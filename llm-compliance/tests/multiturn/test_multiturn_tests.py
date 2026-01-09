"""Test file to evaluate RAG conversational test cases."""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
# File is at: llm-compliance/tests/multiturn/test_multiturn_tests.py
# So parent.parent.parent = llm-compliance/
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.deepEvalMultiTurnMetrics import (
    evaluate_conversation_completeness_test_cases,
    evaluate_knowledge_retention_test_cases,
    evaluate_role_adherence_test_cases,
    load_conversational_test_cases_from_json,
    evaluate_conversational_test_cases
)

load_dotenv()


def evaluate_rag_conversations():
    """Load and evaluate RAG conversational test cases."""
    # Path to the RAG conversations JSON file
    # From llm-compliance/ to utils/synthetic_data/...
    #json_path = project_root / "utils" / "synthetic_data" / "rag_conversations_20251224_211545" / "rag_conversations_20251224_211545.json"
    json_path = project_root / "utils" / "synthetic_data" / "rag_conversations_20251224_215900" / "rag_conversations_20251224_215900.json"
    
    # Debug: Print path information
    print(f"🔍 Debug Info:")
    print(f"   Current file: {Path(__file__)}")
    print(f"   Project root: {project_root}")
    print(f"   JSON path: {json_path}")
    print(f"   JSON path exists: {json_path.exists()}")
    print()
    
    if not json_path.exists():
        print(f"❌ Error: JSON file not found at {json_path}")
        return
    
    print(f"📂 Loading conversational test cases from: {json_path}")
    test_cases = load_conversational_test_cases_from_json(str(json_path))
    print(f"✅ Loaded {len(test_cases)} conversational test case(s)\n")
    
    # Display test case information
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- Test Case {i} ---")
        if hasattr(test_case, 'scenario') and test_case.scenario:
            print(f"Scenario: {test_case.scenario}")
        if hasattr(test_case, 'expected_outcome') and test_case.expected_outcome:
            print(f"Expected Outcome: {test_case.expected_outcome}")
        print(f"Number of Turns: {len(test_case.turns)}")
        print(f"Turns:")
        for j, turn in enumerate(test_case.turns, 1):
            content_preview = turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
            print(f"  {j}. [{turn.role}]: {content_preview}")
        print()
    
    # Run evaluation on all test cases
    print("=" * 80)
    print("🔍 Running Turn Relevancy Evaluation")
    print("=" * 80)
    
    # Evaluate using the utility function
    results = evaluate_knowledge_retention_test_cases(
        test_cases=test_cases,
        threshold=0.5,
        model='gpt-4o',
        include_reason=True
    )
    
    # Print detailed results
    print("\n" + "=" * 80)
    print("📊 Evaluation Results")
    print("=" * 80)
    
    for i, (test_case, score, reason, passed, error_msg) in enumerate(results, 1):
        print(f"\n--- Test Case {i} Results ---")
        if hasattr(test_case, 'scenario') and test_case.scenario:
            print(f"Scenario: {test_case.scenario[:100]}...")
        
        if error_msg:
            print(f"❌ Error: {error_msg}")
        else:
            print(f"Turn Relevancy Score: {score:.4f}" if score is not None else "Turn Relevancy Score: N/A")
            print(f"Passed: {'✅' if passed else '❌'}")
            if reason:
                print(f"Reason: {reason}")
    
    print("\n" + "=" * 80)
    print("✅ Evaluation Complete")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    evaluate_rag_conversations()
