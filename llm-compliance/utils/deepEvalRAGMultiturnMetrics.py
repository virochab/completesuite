"""DeepEval RAG-specific multi-turn conversational metrics verification utilities.

This module contains metrics that require retrieval context, such as TurnFaithfulnessMetric,
TurnContextualPrecisionMetric, TurnContextualRecallMetric, and TurnContextualRelevancyMetric.
"""

import json
import yaml
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from deepeval.metrics import TurnFaithfulnessMetric, TurnContextualPrecisionMetric, TurnContextualRecallMetric, TurnContextualRelevancyMetric
from deepeval.test_case import Turn, ConversationalTestCase
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


def _get_turn_faithfulness_threshold() -> float:
    """Get turn faithfulness threshold from config, default to 0.5."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn_rag', {}).get('turn_faithfulness_min', 0.5)


def _get_turn_faithfulness_model() -> str:
    """Get turn faithfulness model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn_rag', {}).get('turn_faithfulness_model', 'gpt-4o')


def _get_turn_contextual_precision_threshold() -> float:
    """Get turn contextual precision threshold from config, default to 0.5."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn_rag', {}).get('turn_context_precision_min', 0.5)


def _get_turn_contextual_precision_model() -> str:
    """Get turn contextual precision model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn_rag', {}).get('turn_context_precision_model', 'gpt-4o')


def _get_turn_contextual_recall_threshold() -> float:
    """Get turn contextual recall threshold from config, default to 0.5."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn_rag', {}).get('turn_context_recall_min', 0.5)


def _get_turn_contextual_recall_model() -> str:
    """Get turn contextual recall model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn_rag', {}).get('turn_context_recall_model', 'gpt-4o')


def _get_turn_contextual_relevancy_threshold() -> float:
    """Get turn contextual relevancy threshold from config, default to 0.5."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn_rag', {}).get('turn_context_relevance_min', 0.5)


def _get_turn_contextual_relevancy_model() -> str:
    """Get turn contextual relevancy model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn_rag', {}).get('turn_context_relevance_model', 'gpt-4o')


def load_conversational_test_cases_from_json(json_path: str) -> List[ConversationalTestCase]:
    """
    Load ConversationalTestCase objects from a JSON file with retrieval context support.
    
    This function handles retrieval_context for RAG-specific metrics like TurnFaithfulnessMetric,
    TurnContextualPrecisionMetric, TurnContextualRecallMetric, and TurnContextualRelevancyMetric. 
    It includes fallback logic to assign test case-level context to assistant turns if retrieval_context 
    is not present at the turn level (for backward compatibility).
    
    Args:
        json_path: Path to the JSON file containing conversational test cases
    
    Returns:
        List of ConversationalTestCase objects with retrieval_context set on assistant turns
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = []
    for item in data:
        # Convert turns to Turn objects
        # RAG metrics (TurnFaithfulnessMetric, TurnContextualPrecisionMetric, TurnContextualRecallMetric, 
        # TurnContextualRelevancyMetric) require retrieval_context, so include it if present
        # Fallback: if test case has 'context' but turns don't have 'retrieval_context',
        # assign context to assistant turns (for backward compatibility)
        test_case_context = item.get('context', [])
        turns = []
        for turn_data in item.get('turns', []):
            turn_kwargs = {
                'role': turn_data.get('role', 'user'),
                'content': turn_data.get('content', '')
            }
            # Include retrieval_context if present (required for RAG metrics like TurnFaithfulnessMetric, 
            # TurnContextualPrecisionMetric, TurnContextualRecallMetric, TurnContextualRelevancyMetric)
            if 'retrieval_context' in turn_data:
                turn_kwargs['retrieval_context'] = turn_data['retrieval_context']
            elif turn_data.get('role') == 'assistant' and test_case_context:
                # Fallback: use test case level context for assistant turns if available
                turn_kwargs['retrieval_context'] = test_case_context if isinstance(test_case_context, list) else [test_case_context]
            turns.append(Turn(**turn_kwargs))
        
        # Create ConversationalTestCase
        chatbot_role = item.get('chatbot_role', None)
        topics = item.get('topics', None)
        
        # Create test case with appropriate parameters
        kwargs = {}
        if chatbot_role:
            kwargs['chatbot_role'] = chatbot_role
        
        test_case = ConversationalTestCase(turns=turns, **kwargs)
        
        # Add metadata if available
        if 'scenario' in item:
            test_case.scenario = item['scenario']
        if 'expected_outcome' in item:
            test_case.expected_outcome = item['expected_outcome']
        if 'user_description' in item:
            test_case.user_description = item['user_description']
        if 'context' in item:
            test_case.context = item['context']
        if topics:
            test_case.topics = topics
        
        test_cases.append(test_case)
    
    return test_cases


def verify_turn_faithfulness_metric(
    test_case: ConversationalTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify TurnFaithfulnessMetric for a conversational test case.
    
    TurnFaithfulnessMetric evaluates whether the LLM chatbot generates factually accurate responses 
    grounded in the retrieval context throughout a conversation.
    
    Args:
        test_case: The ConversationalTestCase to evaluate (must have retrieval_context in assistant turns)
        threshold: Minimum required faithfulness score. If None, loads from thresholds.yaml (deepeval.multiturn.turn_faithfulness_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.turn_faithfulness_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Turn faithfulness metric score (0-1, higher = more faithful)
            - reason: Explanation of the faithfulness assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    # Validate that assistant turns have retrieval_context
    missing_context = []
    for i, turn in enumerate(test_case.turns):
        # Only require retrieval_context for assistant turns
        if turn.role == "assistant":
            if not hasattr(turn, 'retrieval_context') or not turn.retrieval_context:
                missing_context.append(f"turn {i+1} (assistant)")
    
    if missing_context:
        error_msg = f"ConversationalTestCase must have retrieval_context set for assistant turns in TurnFaithfulnessMetric. Missing in: {', '.join(missing_context)}"
        if case_id:
            error_msg += f" (case {case_id})"
        return None, None, False, error_msg
    
    try:
        # Use config defaults if not provided
        if threshold is None:
            threshold = _get_turn_faithfulness_threshold()
        if model is None:
            model = _get_turn_faithfulness_model()
        
        # Create and measure metric
        metric = TurnFaithfulnessMetric(
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
        error_msg = f"Error evaluating TurnFaithfulnessMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def evaluate_turn_faithfulness_test_cases(
    test_cases: List[ConversationalTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[ConversationalTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of conversational test cases using TurnFaithfulnessMetric.
    
    TurnFaithfulnessMetric evaluates whether the LLM chatbot generates factually accurate responses 
    grounded in the retrieval context throughout a conversation.
    
    Args:
        test_cases: List of ConversationalTestCase objects to evaluate (must have retrieval_context in assistant turns)
        threshold: Minimum required faithfulness score. If None, loads from thresholds.yaml (deepeval.multiturn.turn_faithfulness_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.turn_faithfulness_model)
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_turn_faithfulness_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def verify_turn_contextual_precision_metric(
    test_case: ConversationalTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify TurnContextualPrecisionMetric for a conversational test case.
    
    TurnContextualPrecisionMetric evaluates the precision of the retrieval context used
    in generating the assistant's response. It measures how much of the retrieved context
    is actually relevant and used in the response.
    
    Args:
        test_case: The ConversationalTestCase to evaluate (must have retrieval_context in assistant turns)
        threshold: Minimum required contextual precision score. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_precision_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_precision_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Turn contextual precision metric score (0-1, higher = more precise)
            - reason: Explanation of the contextual precision assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    # Validate that assistant turns have retrieval_context
    missing_context = []
    for i, turn in enumerate(test_case.turns):
        # Only require retrieval_context for assistant turns
        if turn.role == "assistant":
            if not hasattr(turn, 'retrieval_context') or not turn.retrieval_context:
                missing_context.append(f"turn {i+1} (assistant)")
    
    if missing_context:
        error_msg = f"ConversationalTestCase must have retrieval_context set for assistant turns in TurnContextualPrecisionMetric. Missing in: {', '.join(missing_context)}"
        if case_id:
            error_msg += f" (case {case_id})"
        return None, None, False, error_msg
    
    try:
        # Use config defaults if not provided
        if threshold is None:
            threshold = _get_turn_contextual_precision_threshold()
        if model is None:
            model = _get_turn_contextual_precision_model()
        
        # Create and measure metric
        metric = TurnContextualPrecisionMetric(
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
        error_msg = f"Error evaluating TurnContextualPrecisionMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def evaluate_turn_contextual_precision_test_cases(
    test_cases: List[ConversationalTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[ConversationalTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of conversational test cases using TurnContextualPrecisionMetric.
    
    TurnContextualPrecisionMetric evaluates the precision of the retrieval context used
    in generating the assistant's response. It measures how much of the retrieved context
    is actually relevant and used in the response.
    
    Args:
        test_cases: List of ConversationalTestCase objects to evaluate (must have retrieval_context in assistant turns)
        threshold: Minimum required contextual precision score. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_precision_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_precision_model)
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_turn_contextual_precision_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def verify_turn_contextual_recall_metric(
    test_case: ConversationalTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify TurnContextualRecallMetric for a conversational test case.
    
    TurnContextualRecallMetric evaluates the recall of the retrieval context used
    in generating the assistant's response. It measures how much of the relevant information
    from the expected outcome is covered by the retrieved context and used in the response.
    
    Args:
        test_case: The ConversationalTestCase to evaluate (must have retrieval_context in assistant turns)
        threshold: Minimum required contextual recall score. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_recall_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_recall_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Turn contextual recall metric score (0-1, higher = better recall)
            - reason: Explanation of the contextual recall assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    # Validate that assistant turns have retrieval_context
    missing_context = []
    for i, turn in enumerate(test_case.turns):
        # Only require retrieval_context for assistant turns
        if turn.role == "assistant":
            if not hasattr(turn, 'retrieval_context') or not turn.retrieval_context:
                missing_context.append(f"turn {i+1} (assistant)")
            else:
                # Debug: Log retrieval context for troubleshooting
                if case_id:
                    context_preview = str(turn.retrieval_context)[:200] if turn.retrieval_context else "None"
                    content_preview = turn.content[:200] if turn.content else "None"
                    print(f"🔍 Debug [{case_id}] Turn {i+1} (assistant):")
                    print(f"   Content preview: {content_preview}...")
                    print(f"   Retrieval context preview: {context_preview}...")
                    # Check if "False" appears in content or context
                    if "False" in turn.content or "false" in turn.content.lower():
                        print(f"   ⚠️  WARNING: 'False' found in assistant content!")
                    if turn.retrieval_context:
                        context_str = " ".join(turn.retrieval_context) if isinstance(turn.retrieval_context, list) else str(turn.retrieval_context)
                        if "False" in context_str or "false" in context_str.lower():
                            print(f"   ⚠️  WARNING: 'False' found in retrieval context!")
                        else:
                            print(f"   ✅ 'False' NOT found in retrieval context")
    
    if missing_context:
        error_msg = f"ConversationalTestCase must have retrieval_context set for assistant turns in TurnContextualRecallMetric. Missing in: {', '.join(missing_context)}"
        if case_id:
            error_msg += f" (case {case_id})"
        return None, None, False, error_msg
    
    try:
        # Use config defaults if not provided
        if threshold is None:
            threshold = _get_turn_contextual_recall_threshold()
        if model is None:
            model = _get_turn_contextual_recall_model()
        
        # Create and measure metric
        metric = TurnContextualRecallMetric(
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
        error_msg = f"Error evaluating TurnContextualRecallMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def evaluate_turn_contextual_recall_test_cases(
    test_cases: List[ConversationalTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[ConversationalTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of conversational test cases using TurnContextualRecallMetric.
    
    TurnContextualRecallMetric evaluates the recall of the retrieval context used
    in generating the assistant's response. It measures how much of the relevant information
    from the expected outcome is covered by the retrieved context and used in the response.
    
    Args:
        test_cases: List of ConversationalTestCase objects to evaluate (must have retrieval_context in assistant turns)
        threshold: Minimum required contextual recall score. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_recall_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_recall_model)
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_turn_contextual_recall_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def verify_turn_contextual_relevancy_metric(
    test_case: ConversationalTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify TurnContextualRelevancyMetric for a conversational test case.
    
    TurnContextualRelevancyMetric evaluates the relevancy of the retrieval context used
    in generating the assistant's response. It measures how relevant the retrieved context
    is to the user's query and the assistant's response.
    
    Args:
        test_case: The ConversationalTestCase to evaluate (must have retrieval_context in assistant turns)
        threshold: Minimum required contextual relevancy score. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_relevance_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_relevance_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Turn contextual relevancy metric score (0-1, higher = more relevant)
            - reason: Explanation of the contextual relevancy assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    # Validate that assistant turns have retrieval_context
    missing_context = []
    for i, turn in enumerate(test_case.turns):
        # Only require retrieval_context for assistant turns
        if turn.role == "assistant":
            if not hasattr(turn, 'retrieval_context') or not turn.retrieval_context:
                missing_context.append(f"turn {i+1} (assistant)")
    
    if missing_context:
        error_msg = f"ConversationalTestCase must have retrieval_context set for assistant turns in TurnContextualRelevancyMetric. Missing in: {', '.join(missing_context)}"
        if case_id:
            error_msg += f" (case {case_id})"
        return None, None, False, error_msg
    
    try:
        # Use config defaults if not provided
        if threshold is None:
            threshold = _get_turn_contextual_relevancy_threshold()
        if model is None:
            model = _get_turn_contextual_relevancy_model()
        
        # Create and measure metric
        metric = TurnContextualRelevancyMetric(
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
        error_msg = f"Error evaluating TurnContextualRelevancyMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def evaluate_turn_contextual_relevancy_test_cases(
    test_cases: List[ConversationalTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[ConversationalTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of conversational test cases using TurnContextualRelevancyMetric.
    
    TurnContextualRelevancyMetric evaluates the relevancy of the retrieval context used
    in generating the assistant's response. It measures how relevant the retrieved context
    is to the user's query and the assistant's response.
    
    Args:
        test_cases: List of ConversationalTestCase objects to evaluate (must have retrieval_context in assistant turns)
        threshold: Minimum required contextual relevancy score. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_relevance_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn_rag.turn_context_relevance_model)
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_turn_contextual_relevancy_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results

