"""Test RAGAS metrics on single-turn golden test cases.

This test file focuses on RAG-specific metrics using RAGAS library:
- faithfulness: Evaluates factual accuracy grounded in retrieval context
- answer_relevancy: Evaluates how relevant the response is to the query
- context_precision: Evaluates precision of retrieval context usage
- context_recall: Evaluates recall of relevant information from context
- context_entity_recall: Evaluates recall of important entities
- answer_correctness: Evaluates correctness of answers against ground truth
"""

import json
import math
import pytest
from pathlib import Path
import sys
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()
# RAGAS imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_correctness
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.metricsHistoryTracker import MetricsHistoryTracker
from app.clientrag import RAGClient

# Metric name to threshold path mapping (RAGAS metrics)
METRIC_THRESHOLD_MAP = {
    "faithfulness": ("rags", "faithfulness_min"),
    "answer_relevancy": ("rags", "answer_relevancy_min"),
    "context_precision": ("rags", "context_precision_min"),
    "context_recall": ("rags", "context_recall_min"),
    "context_entity_recall": ("rags", "context_entity_recall_min"),
    "answer_correctness": ("rags", "answer_correctness_min"),
}

# Metric name to mean threshold path mapping
METRIC_MEAN_THRESHOLD_MAP = {
    "faithfulness": ("rags", "faithfulness_mean"),
    "answer_relevancy": ("rags", "answer_relevancy_mean"),
    "context_precision": ("rags", "context_precision_mean"),
    "context_recall": ("rags", "context_recall_mean"),
    "context_entity_recall": ("rags", "context_entity_recall_mean"),
    "answer_correctness": ("rags", "answer_correctness_mean"),
}


def load_singleturn_goldens_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load single-turn golden test cases from JSON file.
    
    Single-turn goldens have the structure:
    {
        "input": "user query",
        "actual_output": "assistant response" or null,
        "expected_output": "expected response",
        "context": ["context chunk 1", "context chunk 2", ...],
        "source_file": "path/to/source.pdf"
    }
    
    Args:
        json_path: Path to the JSON file containing single-turn goldens
    
    Returns:
        List of dictionaries with test case data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = []
    
    for item in data:
        input_query = item.get('input', '')
        actual_output = item.get('actual_output')
        expected_output = item.get('expected_output', '')
        context = item.get('context', [])
        source_file = item.get('source_file', '')
        
        # Use actual_output if available, otherwise use expected_output
        answer = actual_output if actual_output else expected_output
        
        if not answer:
            continue  # Skip if no answer
        
        # Ensure context is a list
        if not isinstance(context, list):
            context = [context] if context else []
        
        test_case = {
            'question': input_query,
            'answer': answer,
            'contexts': context,
            'ground_truth': expected_output,
            'source_file': source_file
        }
        
        test_cases.append(test_case)
    
    return test_cases


def get_threshold(thresholds: Dict[str, Any], metric_name: str) -> float:
    """Get threshold value for a metric from nested thresholds structure."""
    if metric_name not in METRIC_THRESHOLD_MAP:
        return 0.7  # Default threshold
    
    path = METRIC_THRESHOLD_MAP[metric_name]
    value = thresholds
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return 0.7  # Default if path not found
    
    return value


def get_metric_mean_threshold(thresholds: Dict[str, Any], metric_name: str) -> float:
    """Get mean threshold value for a metric from nested thresholds structure."""
    if metric_name not in METRIC_MEAN_THRESHOLD_MAP:
        return 0.7  # Default mean threshold
    
    path = METRIC_MEAN_THRESHOLD_MAP[metric_name]
    value = thresholds
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return 0.7  # Default if path not found
    
    return value


def printSaveIndividualResults(
    results: List[Dict[str, Any]],
    test_cases: List[Dict[str, Any]],
    metric_names: List[str],
    reports_dir: Path,
    timestamp: str
) -> Optional[Path]:
    """
    Print and save individual query-level results to CSV.
    
    Args:
        results: List of result entries with metrics for each query
        test_cases: Original test cases with context information
        metric_names: List of metric names being evaluated
        reports_dir: Directory to save the CSV file
        timestamp: Timestamp string for filename
    
    Returns:
        Path to the saved CSV file, or None if no results
    """
    if not results:
        return None
    
    csv_file = reports_dir / f"ragas_rag_metrics_test_results_{timestamp}.csv"
    
    # Flatten results for CSV (one row per test case/query, metrics as columns)
    csv_rows = []
    for i, entry in enumerate(results):
        # Get the original test case to include contexts
        test_case = test_cases[i] if i < len(test_cases) else {}
        contexts = test_case.get('contexts', [])
        
        # Prepare contexts as a single string (truncated if too long)
        contexts_text = ""
        if contexts:
            # Join contexts with a separator, truncate if total length > 2000 chars
            contexts_joined = " | ".join(str(ctx) for ctx in contexts)
            if len(contexts_joined) > 2000:
                contexts_text = contexts_joined[:2000] + "... (truncated)"
            else:
                contexts_text = contexts_joined
        
        row = {
            "case_id": entry.get("case_id"),
            "question": entry.get("question"),
            "answer": entry.get("answer"),
            "ground_truth": entry.get("ground_truth"),
            "source_file": entry.get("source_file"),
            "contexts_count": entry.get("contexts_count"),
            "contexts_preview": contexts_text[:500] if contexts_text else "",  # First 500 chars for preview
            "contexts_full": contexts_text,  # Full contexts (may be truncated)
            "timestamp": entry.get("timestamp"),
        }
        
        # Add metric scores as columns (one column per metric with score, threshold, passed)
        metrics = entry.get("metrics", {})
        for metric_name, metric_data in metrics.items():
            row[f"{metric_name}_score"] = metric_data.get("score")
            row[f"{metric_name}_threshold"] = metric_data.get("threshold")
            row[f"{metric_name}_passed"] = metric_data.get("passed")
            # Also include error if present
            if "error" in metric_data:
                row[f"{metric_name}_error"] = metric_data.get("error")
        
        # Calculate overall pass/fail for this query (all metrics must pass)
        all_passed = all(
            metric_data.get("passed", False) 
            for metric_data in metrics.values() 
            if metric_data.get("score") is not None
        )
        row["overall_passed"] = all_passed if metrics else False
        
        # Count how many metrics passed
        passed_count = sum(1 for m in metrics.values() if m.get("passed", False))
        total_metrics = len([m for m in metrics.values() if m.get("score") is not None])
        row["metrics_passed"] = passed_count
        row["metrics_total"] = total_metrics
        row["metrics_pass_rate"] = round((passed_count / total_metrics * 100) if total_metrics > 0 else 0, 2)
        
        # Add errors if any
        errors = entry.get("errors", [])
        row["errors"] = json.dumps(errors) if errors else None
        
        csv_rows.append(row)
    
    # Get all unique fieldnames
    all_fieldnames = set()
    for row in csv_rows:
        all_fieldnames.update(row.keys())
    
    # Define RAGAS evaluation metrics in order
    ragas_metrics = [
        'faithfulness',
        'answer_relevancy',
        'context_precision',
        'context_recall',
        'context_entity_recall',
        'answer_correctness'
    ]
    
    # Build metric columns in order: for each metric, score, threshold, passed, error
    metric_columns = []
    for metric_name in ragas_metrics:
        # Add columns for this metric if they exist
        if f"{metric_name}_score" in all_fieldnames:
            metric_columns.append(f"{metric_name}_score")
        if f"{metric_name}_threshold" in all_fieldnames:
            metric_columns.append(f"{metric_name}_threshold")
        if f"{metric_name}_passed" in all_fieldnames:
            metric_columns.append(f"{metric_name}_passed")
        if f"{metric_name}_error" in all_fieldnames:
            metric_columns.append(f"{metric_name}_error")
    
    # Define leftmost columns in desired order (query info first, then metrics)
    leftmost_columns = [
        "case_id", 
        "question", 
        "answer", 
        "ground_truth", 
        "source_file", 
        "contexts_count",
        "contexts_preview",
    ]
    
    # Add summary columns
    summary_columns = [
        "overall_passed",
        "metrics_passed",
        "metrics_total",
        "metrics_pass_rate",
    ]
    
    # Add metric columns (individual scores for each query)
    # These are individual query-level scores, not mean scores
    ragas_metric_columns = metric_columns
    
    # Add remaining columns (if any unexpected ones exist)
    remaining_columns = sorted([
        f for f in all_fieldnames 
        if f not in leftmost_columns 
        and f not in summary_columns 
        and f not in ragas_metric_columns
        and f != "errors"
        and f != "timestamp"
        and f != "contexts_full"
    ])
    
    # Final column order: query info -> RAGAS metrics -> summary -> errors -> timestamp
    fieldnames = []
    # Add query info columns
    fieldnames.extend([col for col in leftmost_columns if col in all_fieldnames])
    # Add RAGAS metric columns (individual scores for each query, not mean scores)
    fieldnames.extend(ragas_metric_columns)
    # Add summary columns
    fieldnames.extend([col for col in summary_columns if col in all_fieldnames])
    # Add errors and timestamp if they exist
    if "errors" in all_fieldnames:
        fieldnames.append("errors")
    if "timestamp" in all_fieldnames:
        fieldnames.append("timestamp")
    # Add any remaining columns
    fieldnames.extend(remaining_columns)
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            # Ensure all fields are present
            complete_row = {field: row.get(field, None) for field in fieldnames}
            writer.writerow(complete_row)
    
    # Print results
    print(f"\n📄 Individual Query-Level Results:")
    print(f"   CSV report saved: {csv_file}")
    print(f"   Total queries: {len(csv_rows)}")
    print(f"   Columns: {len(fieldnames)} (including {len(metric_names)} metrics)")
    
    return csv_file


def printSaveSummaryReport(
    results: List[Dict[str, Any]],
    metric_names: List[str],
    metric_averages: Dict[str, Dict[str, Any]],
    metric_mean_failures: List[str],
    test_cases: List[Dict[str, Any]],
    failures: List[str],
    log_file: Path,
    reports_dir: Path,
    timestamp: str
) -> Optional[Path]:
    """
    Print and save summary report with per-metric statistics.
    
    Args:
        results: List of result entries with metrics for each query
        metric_names: List of metric names being evaluated
        metric_averages: Dictionary of metric averages and statistics
        metric_mean_failures: List of failure messages for metrics below mean threshold
        test_cases: Original test cases
        failures: List of individual test case failures
        log_file: Path to JSONL log file
        reports_dir: Directory to save the summary CSV file
        timestamp: Timestamp string for filename
    
    Returns:
        Path to the saved summary CSV file, or None if no metric_averages
    """
    # Log summary before any assertions
    total_test_cases = len(test_cases)
    total_metrics_tested = total_test_cases * len(metric_names)
    total_failures = len(failures)
    
    print(f"\n📊 RAGAS RAG Metrics Test Summary:")
    print(f"  Total test cases: {total_test_cases}")
    print(f"  Total metrics tested: {total_metrics_tested}")
    print(f"  Total failures: {total_failures}")
    print(f"  Log file: {log_file}")
    
    # Calculate and print per-metric summary
    print(f"\n📈 Per-Metric Summary:")
    
    for metric_name in metric_names:
        metric_data = metric_averages.get(metric_name, {})
        if metric_data:
            metric_passed = metric_data.get("passed_count", 0)
            metric_total = metric_data.get("total_count", 0)
            pass_rate = metric_data.get("pass_rate", 0.0)
            avg_score = metric_data.get("average", 0.0)
            mean_threshold = metric_data.get("mean_threshold", 0.0)
            status = metric_data.get("status", "UNKNOWN")
            
            print(f"  {metric_name}:")
            print(f"    Passed: {metric_passed}/{metric_total} ({pass_rate:.1f}%)")
            print(f"    Average Score: {avg_score:.4f}")
            print(f"    Mean Threshold: {mean_threshold:.4f}")
            print(f"    Status: {status}")
            
            # Check if average meets mean threshold
            if not metric_data.get("meets_threshold", False):
                failure_msg = (
                    f"{metric_name} average score {avg_score:.4f} is below mean threshold {mean_threshold:.4f}"
                )
                print(f"    ⚠️  FAILED: {failure_msg}")
            else:
                print(f"    ✅ PASSED: Average meets mean threshold")
    
    # Export metric-level summary to CSV
    if not metric_averages:
        return None
    
    summary_csv_file = reports_dir / f"ragas_rag_metrics_summary_{timestamp}.csv"
    summary_fieldnames = ["metric", "number_of_tests", "mean", "mean_threshold", "status"]
    
    with open(summary_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for metric_name, metric_data in metric_averages.items():
            writer.writerow({
                "metric": metric_name,
                "number_of_tests": metric_data["total_count"],
                "mean": round(metric_data["average"], 4),
                "mean_threshold": round(metric_data["mean_threshold"], 4),
                "status": metric_data["status"]
            })
    
    print(f"\n📄 Summary Report:")
    print(f"   Summary CSV saved: {summary_csv_file}")
    
    return summary_csv_file


def test_ragas_rag_metrics(thresholds):
    """Test RAGAS metrics on single-turn golden test cases.
    
    This test focuses on RAG metrics using RAGAS library:
    - faithfulness: Evaluates factual accuracy grounded in retrieval context
    - answer_relevancy: Evaluates how relevant the response is to the query
    - context_precision: Evaluates precision of retrieval context usage
    - context_recall: Evaluates recall of relevant information from context
    - context_entity_recall: Evaluates recall of important entities
    - answer_correctness: Evaluates correctness of answers against ground truth
    
    The test calls the RAG model for each query to get live answers:
    - question: user query from test data
    - answer: actual answer from RAG model (via RAGClient)
    - contexts: retrieved contexts from RAG response (or test data contexts as fallback)
    - ground_truth: expected_output from test data (for metrics that need it)
    
    Note: The RAG model is called via RAGClient, which tries API first (http://localhost:8000)
    and falls back to direct agent mode if API is unavailable.
    """
    # Load single-turn golden test cases
    project_root = Path(__file__).parent.parent.parent
    json_path = project_root / "testData" / "synthetic_data" / "singleturn" / "singleturnGoldens" / "20251225_232614.json"
    
    if not json_path.exists():
        pytest.skip(f"Single-turn goldens file not found: {json_path}")
    
    # Load test cases
    test_cases = load_singleturn_goldens_from_json(str(json_path))
    
    if not test_cases:
        pytest.skip("No test cases found in file")
    
    # Verify that test cases have contexts for RAG metrics
    for i, test_case in enumerate(test_cases, 1):
        if not test_case.get('contexts') or len(test_case.get('contexts', [])) == 0:
            pytest.skip(f"Test case {i} missing contexts")
    
    # Setup logging directories
    evidence_dir = project_root / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = evidence_dir / f"ragas_rag_metrics_test_log_{timestamp}.jsonl"
    
    # Initialize RAG client to call the RAG model for each query
    # Try API first, fallback to direct agent if API is not available
    print("\n🔄 Initializing RAG client...")
    try:
        rag_client = RAGClient(use_api=True, api_url="http://localhost:8000", api_endpoint="/query")
        print("✅ Using RAG API client")
    except Exception as e:
        print(f"⚠️  Could not initialize API client: {e}")
        print("💡 Falling back to direct agent mode...")
        rag_client = RAGClient(use_api=False)
        print("✅ Using direct RAG agent")
    
    # Call RAG model for each test case to get actual answers
    print(f"\n📞 Calling RAG model for {len(test_cases)} test cases...")
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    rag_errors = []
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case['question']
        ground_truth = test_case['ground_truth']
        expected_contexts = test_case.get('contexts', [])
        
        print(f"  [{i}/{len(test_cases)}] Query: {question[:80]}...")
        
        try:
            # Call RAG model to get actual answer
            rag_response = rag_client.ask(question, debug=False)
            actual_answer = rag_response.text if hasattr(rag_response, 'text') else str(rag_response)
            
            # Extract contexts from RAG response (if available) or use test data contexts
            # RAGResponse has both 'context' (list of strings) and 'citations' (list of dicts with 'content')
            retrieved_contexts = None
            if hasattr(rag_response, 'context') and rag_response.context:
                # Use context attribute (list of context chunks)
                retrieved_contexts = rag_response.context
            elif hasattr(rag_response, 'citations') and rag_response.citations:
                # Extract from citations if context not available
                retrieved_contexts = [citation.get('content', '') for citation in rag_response.citations if citation.get('content')]
            
            if retrieved_contexts:
                contexts.append(retrieved_contexts)
            else:
                # Fallback to test data contexts if RAG didn't return contexts
                contexts.append(expected_contexts)
            
            questions.append(question)
            answers.append(actual_answer)
            ground_truths.append(ground_truth)
            
            print(f"      ✅ Got answer ({len(actual_answer)} chars, {len(retrieved_contexts) if retrieved_contexts else len(expected_contexts)} contexts)")
            
        except Exception as e:
            error_msg = f"Error calling RAG model for test case {i}: {str(e)}"
            print(f"      ❌ {error_msg}")
            rag_errors.append(error_msg)
            
            # Use test data as fallback if RAG call fails
            questions.append(question)
            answers.append(test_case.get('answer', ''))  # Use pre-generated answer as fallback
            contexts.append(expected_contexts)
            ground_truths.append(ground_truth)
    
    if rag_errors:
        print(f"\n⚠️  Warning: {len(rag_errors)} RAG call(s) failed:")
        for error in rag_errors[:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(rag_errors) > 5:
            print(f"   ... and {len(rag_errors) - 5} more errors")
    
    print(f"\n✅ Successfully retrieved {len(answers)} answers from RAG model")
    
    # Create dataset for RAGAS evaluation
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(eval_data)
    
    # Wrap embeddings for RAGAS compatibility
    # Use OpenAI embeddings for RAGAS as it's more compatible
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            request_timeout=360,  # Increase timeout to 360 seconds (6 minutes)
            max_retries=3,  # Retry up to 3 times on failure
        )
    )
    
    # Configure LLM for RAGAS metrics evaluation with max_tokens to prevent token limit errors
    # RAGAS metrics use LLMs internally, so we need to configure max_tokens
    # This prevents "The output is incomplete due to a max_tokens length limit" errors
    # 
    # You can set max_tokens in two ways:
    # 1. Pass llm to each metric (if supported by your RAGAS version)
    # 2. Configure via environment variable or RAGAS global settings
    #
    # For now, we'll configure the LLM and try to pass it to metrics
    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o",
            max_tokens=8192,  # Set max_tokens to prevent "incomplete output" errors
            temperature=0.0,
            request_timeout=360,
            max_retries=3,
        )
    )
    
    # Define metrics to evaluate
    # Note: RAGAS version compatibility - some versions accept llm parameter, others don't
    # If you get errors about llm parameter, remove it and configure LLM globally
    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
        answer_correctness
    ]
    
    # Try to configure metrics with LLM if supported
    # Uncomment and modify if your RAGAS version supports it:
    # metrics_to_evaluate = [
    #     faithfulness(llm=ragas_llm),
    #     answer_relevancy(llm=ragas_llm),
    #     context_precision(llm=ragas_llm),
    #     context_recall(llm=ragas_llm),
    #     context_entity_recall(llm=ragas_llm),
    #     answer_correctness(llm=ragas_llm)
    # ]
    
    # Evaluate using RAGAS metrics with error handling
    print("Note: RAGAS evaluation may take several minutes. Timeout set to 360 seconds (6 minutes) per request.")
    print("Note: To set max_tokens, configure the LLM used by RAGAS metrics.")
    print("      You can set OPENAI_MAX_TOKENS environment variable or configure LLM per metric.")
    try:
        # Try passing llm to evaluate (if RAGAS version supports it)
        # Some RAGAS versions don't support llm parameter, so we'll try and fallback
        try:
            score = evaluate(
                dataset,
                metrics=metrics_to_evaluate,
                embeddings=ragas_embeddings,
                llm=ragas_llm  # Pass LLM with max_tokens configuration
            )
        except TypeError as te:
            # If llm parameter is not supported, try without it
            if "llm" in str(te).lower() or "unexpected keyword" in str(te).lower():
                print("⚠️  Warning: RAGAS version doesn't support llm parameter, using default LLM configuration")
                score = evaluate(
                    dataset,
                    metrics=metrics_to_evaluate,
                    embeddings=ragas_embeddings
                )
            else:
                raise
    except TimeoutError as e:
        error_msg = f"RAGAS evaluation timed out: {e}"
        print(f"❌ {error_msg}")
        print("💡 Try: Increasing timeout, reducing number of metrics, or checking network connection")
        pytest.fail(error_msg, pytrace=False)
    except Exception as e:
        error_msg = f"Error during RAGAS evaluation: {e}"
        print(f"❌ {error_msg}")
        print("💡 Check your OpenAI API key and network connection")
        print(f"   Error type: {type(e).__name__}")
        pytest.fail(error_msg, pytrace=False)
    
    # Convert results to pandas DataFrame
    score_df = score.to_pandas()
    
    # Debug: Print DataFrame info
    print(f"\n📊 RAGAS Results DataFrame Info:")
    print(f"   Shape: {score_df.shape}")
    print(f"   Columns: {list(score_df.columns)}")
    if len(score_df) > 0:
        print(f"   First row sample:")
        for col in score_df.columns[:5]:  # Show first 5 columns
            print(f"     {col}: {score_df.iloc[0].get(col)}")
    
    # Check if RAGAS provides explanation/reason columns
    # Some metrics like faithfulness may include explanation fields
    explanation_columns = [col for col in score_df.columns if 'explanation' in col.lower() or 'reason' in col.lower() or 'justification' in col.lower()]
    if explanation_columns:
        print(f"   📝 Found explanation columns: {explanation_columns}")
    else:
        print(f"   ℹ️  No explanation columns found in DataFrame")
        print(f"      Note: RAGAS may provide explanations for some metrics (e.g., faithfulness)")
        print(f"      but they may not be included in the standard DataFrame output")
    
    # Store explanation columns for later use
    metric_explanation_map = {}
    for col in explanation_columns:
        # Extract metric name from column (e.g., 'faithfulness_explanation' -> 'faithfulness')
        for metric in expected_metrics:
            if metric in col.lower():
                metric_explanation_map[metric] = col
                break
    
    # Collect all failures and results
    failures = []
    results = []
    
    # Define expected dataset column names (RAGAS may use different names)
    dataset_columns = [
        'question', 'answer', 'contexts', 'ground_truth',  # Our dataset format
        'user_input', 'retrieved_contexts', 'response', 'reference'  # Alternative RAGAS format
    ]
    
    # Define the actual metrics we're evaluating (explicit list to avoid processing dataset columns)
    expected_metrics = [
        'faithfulness',
        'answer_relevancy',
        'context_precision',
        'context_recall',
        'context_entity_recall',
        'answer_correctness'
    ]
    
    # Get metric names from the score DataFrame columns
    # Only include columns that are actual metrics (not dataset columns)
    all_columns = set(score_df.columns)
    metric_names = [col for col in all_columns if col not in dataset_columns and col in expected_metrics]
    
    # Warn if we find unexpected columns
    unexpected_metrics = [col for col in all_columns if col not in dataset_columns and col not in expected_metrics]
    if unexpected_metrics:
        print(f"⚠️  Warning: Found unexpected metric columns: {unexpected_metrics}")
        print(f"   These will be ignored. Expected metrics: {expected_metrics}")
    
    # Warn if expected metrics are missing
    missing_metrics = [col for col in expected_metrics if col not in all_columns]
    if missing_metrics:
        print(f"⚠️  Warning: Expected metrics not found in results: {missing_metrics}")
    
    print(f"📊 Processing {len(metric_names)} metrics: {metric_names}")
    
    # Validate we have metrics to process
    if not metric_names:
        error_msg = "No valid metrics found in RAGAS results. Check if evaluation completed successfully."
        print(f"❌ {error_msg}")
        pytest.fail(error_msg)
    
    # Process each test case result
    for i, row in score_df.iterrows():
        case_id = f"case_{i+1}"
        
        # Validate index bounds
        if i >= len(test_cases):
            print(f"⚠️  Warning: No test case data for row {i+1}, skipping")
            continue
        
        # Get test case metadata
        test_case = test_cases[i]
        source_file = test_case.get('source_file', None)
        
        # Extract question, answer, ground_truth from row (handle both column name formats)
        question = row.get('question') or row.get('user_input') or test_case.get('question', '')
        answer = row.get('answer') or row.get('response') or test_case.get('answer', '')
        ground_truth = row.get('ground_truth') or row.get('reference') or test_case.get('ground_truth', '')
        
        # Initialize result entry
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "case_id": case_id,
            "question": question if isinstance(question, str) else str(question) if question is not None else '',
            "answer": answer if isinstance(answer, str) else str(answer) if answer is not None else '',
            "ground_truth": ground_truth if isinstance(ground_truth, str) else str(ground_truth) if ground_truth is not None else '',
            "source_file": source_file,
            "contexts_count": len(test_case.get('contexts', [])),
            "metrics": {},
            "errors": []
        }
        
        # Process each metric for this test case
        for metric_name in metric_names:
            try:
                threshold = get_threshold(thresholds, metric_name)
                score_value = row.get(metric_name)
                
                # Handle NaN, None, or invalid values
                if score_value is None or (isinstance(score_value, float) and math.isnan(score_value)):
                    error_msg = f"{metric_name} returned None/NaN score for {case_id}"
                    result_entry["errors"].append(error_msg)
                    result_entry["metrics"][metric_name] = {
                        "score": None,
                        "threshold": threshold,
                        "passed": False,
                        "error": "Metric returned None/NaN score"
                    }
                    failures.append(error_msg)
                else:
                    try:
                        # Ensure score is a float
                        score_value = float(score_value)
                        # Check if it's still NaN after conversion
                        if math.isnan(score_value):
                            raise ValueError("Score is NaN")
                        passed = score_value >= threshold
                    except (ValueError, TypeError) as e:
                        error_msg = f"{metric_name} has invalid score value for {case_id}: {score_value} ({type(score_value).__name__})"
                        result_entry["errors"].append(error_msg)
                        result_entry["metrics"][metric_name] = {
                            "score": None,
                            "threshold": threshold,
                            "passed": False,
                            "error": f"Invalid score value: {str(e)}"
                        }
                        failures.append(error_msg)
                        continue
                    
                    # Check if there's an explanation for this metric
                    explanation = None
                    if metric_name in metric_explanation_map:
                        explanation_col = metric_explanation_map[metric_name]
                        explanation = row.get(explanation_col)
                        if explanation and not isinstance(explanation, str):
                            explanation = str(explanation) if explanation is not None else None
                    
                    metric_result = {
                        "score": round(score_value, 4),
                        "threshold": threshold,
                        "passed": passed
                    }
                    # Add explanation if available
                    if explanation:
                        metric_result["explanation"] = explanation
                    
                    result_entry["metrics"][metric_name] = metric_result
                    
                    # Check if metric passed
                    if not passed:
                        failure_msg = f"{metric_name} score={score_value:.4f} < threshold {threshold} for {case_id}"
                        failures.append(failure_msg)
                        
                        # Log very low scores for debugging
                        if score_value == 0.0:
                            print(f"⚠️  Warning: {metric_name} scored 0.0 for {case_id}")
            
            except Exception as e:
                error_msg = f"Error processing {metric_name} for {case_id}: {e}"
                result_entry["errors"].append(error_msg)
                result_entry["metrics"][metric_name] = {
                    "score": None,
                    "threshold": get_threshold(thresholds, metric_name),
                    "passed": False,
                    "error": str(e)
                }
                failures.append(error_msg)
        
        # Add result entry to results
        results.append(result_entry)
        
        # Log to JSONL file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry) + "\n")
    
    # Save and print individual query-level results
    csv_file = printSaveIndividualResults(
        results=results,
        test_cases=test_cases,
        metric_names=metric_names,
        reports_dir=reports_dir,
        timestamp=timestamp
    )
    
    # Calculate per-metric averages for summary report
    metric_averages = {}
    metric_mean_failures = []
    
    for metric_name in metric_names:
        metric_passed = sum(
            1 for entry in results
            if entry.get("metrics", {}).get(metric_name, {}).get("passed", False)
        )
        metric_total = sum(
            1 for entry in results
            if metric_name in entry.get("metrics", {})
        )
        
        # Calculate average score for this metric
        metric_scores = []
        for entry in results:
            metric_data = entry.get("metrics", {}).get(metric_name, {})
            score = metric_data.get("score")
            if score is not None:
                metric_scores.append(score)
        
        if metric_total > 0:
            pass_rate = (metric_passed / metric_total) * 100
            avg_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
            
            # Get mean threshold for this metric
            mean_threshold = get_metric_mean_threshold(thresholds, metric_name)
            
            # Check if average meets mean threshold
            meets_threshold = avg_score >= mean_threshold
            status = "PASS" if meets_threshold else "FAIL"
            
            metric_averages[metric_name] = {
                "average": avg_score,
                "mean_threshold": mean_threshold,
                "passed_count": metric_passed,
                "total_count": metric_total,
                "pass_rate": pass_rate,
                "status": status,
                "meets_threshold": meets_threshold
            }
            
            # Check if average meets mean threshold
            if not meets_threshold:
                failure_msg = (
                    f"{metric_name} average score {avg_score:.4f} is below mean threshold {mean_threshold:.4f}"
                )
                metric_mean_failures.append(failure_msg)
    
    # Save and print summary report
    summary_csv_file = printSaveSummaryReport(
        results=results,
        metric_names=metric_names,
        metric_averages=metric_averages,
        metric_mean_failures=metric_mean_failures,
        test_cases=test_cases,
        failures=failures,
        log_file=log_file,
        reports_dir=reports_dir,
        timestamp=timestamp
    )
    
    # Track metrics in history for CI/CD trend analysis
    if metric_averages:
        try:
            tracker = MetricsHistoryTracker()
            tracker.append_metrics(
                test_suite="ragas_rag",
                metric_averages=metric_averages,
                run_id=timestamp
            )
            print(f"✅ Metrics tracked in history for trend analysis")
        except Exception as e:
            print(f"⚠️  Warning: Failed to track metrics in history: {e}")
            # Don't fail the test if history tracking fails
    
    # Assert on mean threshold failures at metric level
    if metric_mean_failures:
        failure_summary = (
            f"\n\nFound {len(metric_mean_failures)} metric(s) with average scores below mean threshold:\n" +
            "\n".join(f"  - {f}" for f in metric_mean_failures)
        )
        pytest.fail(failure_summary, pytrace=False)
    
    # Assert individual test case failures at the end (if any)
    if failures:
        failure_message = (
            f"\n\nFound {len(failures)} individual test case failure(s):\n" + 
            "\n".join(f"  - {f}" for f in failures[:50])  # Show first 50 failures
        )
        if len(failures) > 50:
            failure_message += f"\n  ... and {len(failures) - 50} more failures"
        pytest.fail(failure_message, pytrace=False)
