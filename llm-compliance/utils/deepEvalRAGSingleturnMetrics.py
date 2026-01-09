"""DeepEval RAG-specific single-turn metrics verification utilities.

This module contains utilities for evaluating RAG metrics on single-turn Q&A pairs
that require retrieval context, such as AnswerRelevancyMetric, FaithfulnessMetric,
ContextualPrecisionMetric, ContextualRecallMetric, and ContextualRelevancyMetric.
"""

import json
import yaml
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# Cache for loaded thresholds
_thresholds_cache: Optional[Dict[str, Any]] = None


def _load_thresholds() -> Dict[str, Any]:
    """
    Load thresholds from thresholds.yaml config file.
    
    Returns:
        Dictionary containing thresholds, or empty dict if file not found
    """
    global _thresholds_cache
    
    if _thresholds_cache is not None:
        return _thresholds_cache
    
    # Path from utils/ to tests/config/thresholds.yaml
    thresholds_path = Path(__file__).parent.parent / "tests" / "config" / "thresholds.yaml"
    
    try:
        if thresholds_path.exists():
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                _thresholds_cache = yaml.safe_load(f) or {}
        else:
            _thresholds_cache = {}
    except Exception:
        _thresholds_cache = {}
    
    return _thresholds_cache


def _get_answer_relevancy_threshold() -> float:
    """Get answer relevancy threshold from config, default to 0.7."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('singleturn_rag', {}).get('answer_relevancy_min', 0.7)


def _get_answer_relevancy_model() -> str:
    """Get answer relevancy model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('singleturn_rag', {}).get('answer_relevancy_model', 'gpt-4o')


def _get_faithfulness_threshold() -> float:
    """Get faithfulness threshold from config, default to 0.7."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('singleturn_rag', {}).get('faithfulness_min', 0.7)


def _get_faithfulness_model() -> str:
    """Get faithfulness model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('singleturn_rag', {}).get('faithfulness_model', 'gpt-4o')


def _get_contextual_precision_threshold() -> float:
    """Get contextual precision threshold from config, default to 0.7."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('singleturn_rag', {}).get('contextual_precision_min', 0.7)


def _get_contextual_precision_model() -> str:
    """Get contextual precision model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('singleturn_rag', {}).get('contextual_precision_model', 'gpt-4o')


def _get_contextual_recall_threshold() -> float:
    """Get contextual recall threshold from config, default to 0.7."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('singleturn_rag', {}).get('contextual_recall_min', 0.7)


def _get_contextual_recall_model() -> str:
    """Get contextual recall model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('singleturn_rag', {}).get('contextual_recall_model', 'gpt-4o')


def _get_contextual_relevancy_threshold() -> float:
    """Get contextual relevancy threshold from config, default to 0.7."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('singleturn_rag', {}).get('contextual_relevancy_min', 0.7)


def _get_contextual_relevancy_model() -> str:
    """Get contextual relevancy model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('singleturn_rag', {}).get('contextual_relevancy_model', 'gpt-4o')


def load_singleturn_goldens_from_json(json_path: str) -> Tuple[List[LLMTestCase], List[Dict[str, Any]]]:
    """
    Load LLMTestCase objects from a single-turn golden JSON file.
    
    Single-turn goldens have the structure:
    {
        "input": "user query",
        "actual_output": "assistant response" or null,
        "expected_output": "expected response",
        "context": ["context chunk 1", "context chunk 2", ...],
        "source_file": "path/to/source.pdf"
    }
    
    This function converts single-turn goldens to LLMTestCase format with:
    - input: user query
    - actual_output: actual_output if available, otherwise expected_output
    - expected_output: expected_output (for metrics that need it)
    - retrieval_context: context (for RAG metrics)
    
    Args:
        json_path: Path to the JSON file containing single-turn goldens
    
    Returns:
        Tuple of (List of LLMTestCase objects with retrieval_context set, List of metadata dicts)
        Metadata dicts contain: source_file, original_input, original_expected_output
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = []
    metadata_list = []
    
    for item in data:
        input_query = item.get('input', '')
        actual_output = item.get('actual_output')
        expected_output = item.get('expected_output', '')
        context = item.get('context', [])
        source_file = item.get('source_file', '')
        
        # Use actual_output if available, otherwise use expected_output
        output = actual_output if actual_output else expected_output
        
        if not output:
            continue  # Skip if no output
        
        # Ensure context is a list
        if not isinstance(context, list):
            context = [context] if context else []
        
        # Create LLMTestCase
        test_case_kwargs = {
            'input': input_query,
            'actual_output': output
        }
        
        # Add expected_output if available (needed for ContextualPrecisionMetric and ContextualRecallMetric)
        if expected_output:
            test_case_kwargs['expected_output'] = expected_output
        
        # Add retrieval_context if available (needed for RAG metrics)
        if context:
            test_case_kwargs['retrieval_context'] = context
        
        test_case = LLMTestCase(**test_case_kwargs)
        test_cases.append(test_case)
        
        # Store metadata separately (LLMTestCase doesn't support source_file)
        metadata_list.append({
            'source_file': source_file,
            'original_input': input_query,
            'original_expected_output': expected_output
        })
    
    return test_cases, metadata_list


def verify_answer_relevancy_metric(
    test_case: LLMTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify AnswerRelevancyMetric for a single-turn test case.
    
    AnswerRelevancyMetric evaluates how relevant the LLM's response is to the user's query.
    
    Args:
        test_case: The LLMTestCase to evaluate
        threshold: Minimum required relevancy score. If None, loads from thresholds.yaml (deepeval.singleturn_rag.answer_relevancy_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.singleturn_rag.answer_relevancy_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Answer relevancy metric score (0-1, higher = more relevant)
            - reason: Explanation of the relevancy assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    try:
        # Use config defaults if not provided
        if threshold is None:
            threshold = _get_answer_relevancy_threshold()
        if model is None:
            model = _get_answer_relevancy_model()
        
        # Create and measure metric
        metric = AnswerRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
        
        metric.measure(test_case)
        
        score = metric.score
        reason = metric.reason if include_reason else None
        passed = score >= threshold
        
        return score, reason, passed, None
        
    except Exception as e:
        error_msg = f"Error evaluating AnswerRelevancyMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def verify_faithfulness_metric(
    test_case: LLMTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify FaithfulnessMetric for a single-turn test case.
    
    FaithfulnessMetric evaluates whether the LLM's response is factually accurate and
    grounded in the retrieval context.
    
    Args:
        test_case: The LLMTestCase to evaluate (must have retrieval_context)
        threshold: Minimum required faithfulness score. If None, loads from thresholds.yaml (deepeval.singleturn_rag.faithfulness_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.singleturn_rag.faithfulness_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Faithfulness metric score (0-1, higher = more faithful)
            - reason: Explanation of the faithfulness assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    # Validate that test case has retrieval_context
    if not hasattr(test_case, 'retrieval_context') or not test_case.retrieval_context:
        error_msg = f"LLMTestCase must have retrieval_context set for FaithfulnessMetric"
        if case_id:
            error_msg += f" (case {case_id})"
        return None, None, False, error_msg
    
    try:
        # Use config defaults if not provided
        if threshold is None:
            threshold = _get_faithfulness_threshold()
        if model is None:
            model = _get_faithfulness_model()
        
        # Create and measure metric
        metric = FaithfulnessMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
        
        metric.measure(test_case)
        
        score = metric.score
        reason = metric.reason if include_reason else None
        passed = score >= threshold
        
        return score, reason, passed, None
        
    except Exception as e:
        error_msg = f"Error evaluating FaithfulnessMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def verify_contextual_precision_metric(
    test_case: LLMTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify ContextualPrecisionMetric for a single-turn test case.
    
    ContextualPrecisionMetric evaluates the precision of the retrieval context used
    in generating the LLM's response. It measures how much of the retrieved context
    is actually relevant and used in the response.
    
    Args:
        test_case: The LLMTestCase to evaluate (must have retrieval_context and expected_output)
        threshold: Minimum required contextual precision score. If None, loads from thresholds.yaml (deepeval.singleturn_rag.contextual_precision_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.singleturn_rag.contextual_precision_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Contextual precision metric score (0-1, higher = more precise)
            - reason: Explanation of the contextual precision assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    # Validate that test case has retrieval_context and expected_output
    if not hasattr(test_case, 'retrieval_context') or not test_case.retrieval_context:
        error_msg = f"LLMTestCase must have retrieval_context set for ContextualPrecisionMetric"
        if case_id:
            error_msg += f" (case {case_id})"
        return None, None, False, error_msg
    
    if not hasattr(test_case, 'expected_output') or not test_case.expected_output:
        error_msg = f"LLMTestCase must have expected_output set for ContextualPrecisionMetric"
        if case_id:
            error_msg += f" (case {case_id})"
        return None, None, False, error_msg
    
    try:
        # Use config defaults if not provided
        if threshold is None:
            threshold = _get_contextual_precision_threshold()
        if model is None:
            model = _get_contextual_precision_model()
        
        # Create and measure metric
        metric = ContextualPrecisionMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
        
        metric.measure(test_case)
        
        score = metric.score
        reason = metric.reason if include_reason else None
        passed = score >= threshold
        
        return score, reason, passed, None
        
    except Exception as e:
        error_msg = f"Error evaluating ContextualPrecisionMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def verify_contextual_recall_metric(
    test_case: LLMTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify ContextualRecallMetric for a single-turn test case.
    
    ContextualRecallMetric evaluates the recall of the retrieval context used
    in generating the LLM's response. It measures how much of the relevant information
    from the expected outcome is covered by the retrieved context and used in the response.
    
    Args:
        test_case: The LLMTestCase to evaluate (must have retrieval_context and expected_output)
        threshold: Minimum required contextual recall score. If None, loads from thresholds.yaml (deepeval.singleturn_rag.contextual_recall_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.singleturn_rag.contextual_recall_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Contextual recall metric score (0-1, higher = better recall)
            - reason: Explanation of the contextual recall assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    # Validate that test case has retrieval_context and expected_output
    if not hasattr(test_case, 'retrieval_context') or not test_case.retrieval_context:
        error_msg = f"LLMTestCase must have retrieval_context set for ContextualRecallMetric"
        if case_id:
            error_msg += f" (case {case_id})"
        return None, None, False, error_msg
    
    if not hasattr(test_case, 'expected_output') or not test_case.expected_output:
        error_msg = f"LLMTestCase must have expected_output set for ContextualRecallMetric"
        if case_id:
            error_msg += f" (case {case_id})"
        return None, None, False, error_msg
    
    try:
        # Use config defaults if not provided
        if threshold is None:
            threshold = _get_contextual_recall_threshold()
        if model is None:
            model = _get_contextual_recall_model()
        
        # Create and measure metric
        metric = ContextualRecallMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
        
        metric.measure(test_case)
        
        score = metric.score
        reason = metric.reason if include_reason else None
        passed = score >= threshold
        
        return score, reason, passed, None
        
    except Exception as e:
        error_msg = f"Error evaluating ContextualRecallMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def verify_contextual_relevancy_metric(
    test_case: LLMTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify ContextualRelevancyMetric for a single-turn test case.
    
    ContextualRelevancyMetric evaluates the relevancy of the retrieval context used
    in generating the LLM's response. It measures how relevant the retrieved context
    is to the user's query and the LLM's response.
    
    Args:
        test_case: The LLMTestCase to evaluate (must have retrieval_context)
        threshold: Minimum required contextual relevancy score. If None, loads from thresholds.yaml (deepeval.singleturn_rag.contextual_relevancy_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.singleturn_rag.contextual_relevancy_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Contextual relevancy metric score (0-1, higher = more relevant)
            - reason: Explanation of the contextual relevancy assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    # Validate that test case has retrieval_context
    if not hasattr(test_case, 'retrieval_context') or not test_case.retrieval_context:
        error_msg = f"LLMTestCase must have retrieval_context set for ContextualRelevancyMetric"
        if case_id:
            error_msg += f" (case {case_id})"
        return None, None, False, error_msg
    
    try:
        # Use config defaults if not provided
        if threshold is None:
            threshold = _get_contextual_relevancy_threshold()
        if model is None:
            model = _get_contextual_relevancy_model()
        
        # Create and measure metric
        metric = ContextualRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
        
        metric.measure(test_case)
        
        score = metric.score
        reason = metric.reason if include_reason else None
        passed = score >= threshold
        
        return score, reason, passed, None
        
    except Exception as e:
        error_msg = f"Error evaluating ContextualRelevancyMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def evaluate_answer_relevancy_test_cases(
    test_cases: List[LLMTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[LLMTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of test cases using AnswerRelevancyMetric.
    
    Args:
        test_cases: List of LLMTestCase objects to evaluate
        threshold: Minimum required relevancy score. If None, loads from thresholds.yaml
        model: Model to use for evaluation. If None, loads from thresholds.yaml
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_answer_relevancy_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def evaluate_faithfulness_test_cases(
    test_cases: List[LLMTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[LLMTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of test cases using FaithfulnessMetric.
    
    Args:
        test_cases: List of LLMTestCase objects to evaluate (must have retrieval_context)
        threshold: Minimum required faithfulness score. If None, loads from thresholds.yaml
        model: Model to use for evaluation. If None, loads from thresholds.yaml
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_faithfulness_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def evaluate_contextual_precision_test_cases(
    test_cases: List[LLMTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[LLMTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of test cases using ContextualPrecisionMetric.
    
    Args:
        test_cases: List of LLMTestCase objects to evaluate (must have retrieval_context and expected_output)
        threshold: Minimum required contextual precision score. If None, loads from thresholds.yaml
        model: Model to use for evaluation. If None, loads from thresholds.yaml
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_contextual_precision_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def evaluate_contextual_recall_test_cases(
    test_cases: List[LLMTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[LLMTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of test cases using ContextualRecallMetric.
    
    Args:
        test_cases: List of LLMTestCase objects to evaluate (must have retrieval_context and expected_output)
        threshold: Minimum required contextual recall score. If None, loads from thresholds.yaml
        model: Model to use for evaluation. If None, loads from thresholds.yaml
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_contextual_recall_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def evaluate_contextual_relevancy_test_cases(
    test_cases: List[LLMTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[LLMTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of test cases using ContextualRelevancyMetric.
    
    Args:
        test_cases: List of LLMTestCase objects to evaluate (must have retrieval_context)
        threshold: Minimum required contextual relevancy score. If None, loads from thresholds.yaml
        model: Model to use for evaluation. If None, loads from thresholds.yaml
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_contextual_relevancy_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results
