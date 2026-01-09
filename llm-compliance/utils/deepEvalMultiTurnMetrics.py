"""DeepEval multi-turn conversational metrics verification utilities."""

import json
import yaml
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from deepeval.metrics import TurnRelevancyMetric, RoleAdherenceMetric, KnowledgeRetentionMetric, ConversationCompletenessMetric, GoalAccuracyMetric, TopicAdherenceMetric
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
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


def _get_turn_relevancy_threshold() -> float:
    """Get turn relevancy threshold from config, default to 0.7."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('turn_relevancy_min', 0.7)


def _get_turn_relevancy_model() -> str:
    """Get turn relevancy model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('turn_relevancy_model', 'gpt-4o')


def _get_role_adherence_threshold() -> float:
    """Get role adherence threshold from config, default to 0.7."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('role_adherence_min', 0.7)


def _get_role_adherence_model() -> str:
    """Get role adherence model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('role_adherence_model', 'gpt-4o')


def _get_knowledge_retention_threshold() -> float:
    """Get knowledge retention threshold from config, default to 0.7."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('knowledge_retention_min', 0.7)


def _get_knowledge_retention_model() -> str:
    """Get knowledge retention model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('knowledge_retention_model', 'gpt-4o')


def _get_conversation_completeness_threshold() -> float:
    """Get conversation completeness threshold from config, default to 0.7."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('conversation_completeness_min', 0.7)


def _get_conversation_completeness_model() -> str:
    """Get conversation completeness model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('conversation_completeness_model', 'gpt-4o')


def _get_goal_accuracy_threshold() -> float:
    """Get goal accuracy threshold from config, default to 0.5."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('goal_accuracy_min', 0.5)


def _get_goal_accuracy_model() -> str:
    """Get goal accuracy model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('goal_accuracy_model', 'gpt-4o')


def _get_topic_adherence_threshold() -> float:
    """Get topic adherence threshold from config, default to 0.5."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('topic_adherence_metric_min', 0.5)


def _get_topic_adherence_model() -> str:
    """Get topic adherence model from config, default to 'gpt-4o'."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('topic_adherence_metric_model', 'gpt-4o')


def _get_topic_adherence_topics() -> List[str]:
    """Get topic adherence topics from config, default to empty list."""
    thresholds = _load_thresholds()
    return thresholds.get('deepeval', {}).get('multiturn', {}).get('topic_adherence_topics', [])


def load_conversational_test_cases_from_json(json_path: str) -> List[ConversationalTestCase]:
    """
    Load ConversationalTestCase objects from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing conversational test cases
    
    Returns:
        List of ConversationalTestCase objects
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = []
    for item in data:
        # Convert turns to Turn objects
        turns = [
            Turn(role=turn.get('role', 'user'), content=turn.get('content', ''))
            for turn in item.get('turns', [])
        ]
        
        # Create ConversationalTestCase
        # RoleAdherenceMetric requires chatbot_role, so we need to handle it
        # TopicAdherenceMetric requires topics, so we need to handle it
        chatbot_role = item.get('chatbot_role', None)
        topics = item.get('topics', None)
        
        # If topics not in JSON, try to load from thresholds.yaml
       
        # Create test case with appropriate parameters
        # Can have both chatbot_role and topics, or just one, or neither
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
        
        test_cases.append(test_case)
    
    return test_cases


def verify_conversational_metric(
    test_case: ConversationalTestCase,
    metric_type: str,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Unified function to verify conversational metrics using DeepEval.
    
    Args:
        test_case: The ConversationalTestCase to evaluate
        metric_type: Type of metric to use. Options: 'turn_relevancy', 'role_adherence', 'knowledge_retention', 'conversation_completeness', 'goal_accuracy', 'topic_adherence'
        threshold: Minimum required score (fails if score < threshold).
                   If None, loads from thresholds.yaml based on metric_type
        model: Model to use for evaluation. If None, loads from thresholds.yaml based on metric_type
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Metric score (0-1, higher = better)
            - reason: Explanation of the assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    # Map metric types to their classes and config getters
    metric_config = {
        'turn_relevancy': {
            'class': TurnRelevancyMetric,
            'threshold_getter': _get_turn_relevancy_threshold,
            'model_getter': _get_turn_relevancy_model,
            'name': 'TurnRelevancyMetric',
            'requires_chatbot_role': False,
        },
        'role_adherence': {
            'class': RoleAdherenceMetric,
            'threshold_getter': _get_role_adherence_threshold,
            'model_getter': _get_role_adherence_model,
            'name': 'RoleAdherenceMetric',
            'requires_chatbot_role': True,
        },
        'knowledge_retention': {
            'class': KnowledgeRetentionMetric,
            'threshold_getter': _get_knowledge_retention_threshold,
            'model_getter': _get_knowledge_retention_model,
            'name': 'KnowledgeRetentionMetric',
            'requires_chatbot_role': False,
        },
        'conversation_completeness': {
            'class': ConversationCompletenessMetric,
            'threshold_getter': _get_conversation_completeness_threshold,
            'model_getter': _get_conversation_completeness_model,
            'name': 'ConversationCompletenessMetric',
            'requires_chatbot_role': False,
        },
        'goal_accuracy': {
            'class': GoalAccuracyMetric,
            'threshold_getter': _get_goal_accuracy_threshold,
            'model_getter': _get_goal_accuracy_model,
            'name': 'GoalAccuracyMetric',
            'requires_chatbot_role': False,
        },
        'topic_adherence': {
            'class': TopicAdherenceMetric,
            'threshold_getter': _get_topic_adherence_threshold,
            'model_getter': _get_topic_adherence_model,
            'name': 'TopicAdherenceMetric',
            'requires_chatbot_role': False,
        }
    }
    
    # Validate metric type
    if metric_type not in metric_config:
        error_msg = f"Invalid metric_type: {metric_type}. Must be one of: {list(metric_config.keys())}"
        if case_id:
            error_msg += f" (case {case_id})"
        return None, None, False, error_msg
    
    config = metric_config[metric_type]
    
    # Debug: Verify which metric is being used
    if case_id:
        print(f"🔍 Debug: Evaluating {config['name']} for {case_id} (metric_type: {metric_type})")
    
    try:
        # Special validation for RoleAdherenceMetric
        if config['requires_chatbot_role']:
            if not hasattr(test_case, 'chatbot_role') or not test_case.chatbot_role:
                error_msg = f"ConversationalTestCase must have chatbot_role set for {config['name']}"
                if case_id:
                    error_msg += f" (case {case_id})"
                return None, None, False, error_msg
        
        # Special validation for TopicAdherenceMetric
        if config.get('requires_topics', False):
            if not hasattr(test_case, 'topics') or not test_case.topics:
                error_msg = f"ConversationalTestCase must have topics set for {config['name']}"
                if case_id:
                    error_msg += f" (case {case_id})"
                return None, None, False, error_msg
        
        # Use config defaults if not provided
        if threshold is None:
            threshold = config['threshold_getter']()
        if model is None:
            model = config['model_getter']()
        
        # Create and measure metric
        metric_class = config['class']
        # Debug: Verify metric class name
        if case_id:
            print(f"🔍 Debug: Instantiating metric class: {metric_class.__name__}")
        
        # Special handling for TopicAdherenceMetric - requires relevant_topics parameter
        if metric_type == 'topic_adherence':
            # Get topics from test_case or thresholds.yaml
            topics = getattr(test_case, 'topics', None)
            if not topics:
                topics = _get_topic_adherence_topics()
            
            # Validate topics
            if not topics or not isinstance(topics, list) or len(topics) == 0:
                error_msg = f"ConversationalTestCase must have a non-empty list of topics for TopicAdherenceMetric"
                if case_id:
                    error_msg += f" (case {case_id})"
                return None, None, False, error_msg
            
            if case_id:
                print(f"🔍 Debug: Instantiating TopicAdherenceMetric with topics: {topics}")
            
            # TopicAdherenceMetric requires relevant_topics as first positional argument
            metric = TopicAdherenceMetric(
                topics,  # First positional argument: relevant_topics (List[str])
                threshold=threshold,
                model=model,
                include_reason=include_reason
            )
        else:
            # For other metrics, use standard instantiation
            metric = metric_class(
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
        error_msg = f"Error evaluating {config['name']}"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def verify_turn_relevancy_metric(
    test_case: ConversationalTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify turn relevancy using DeepEval TurnRelevancyMetric.
    
    Args:
        test_case: The ConversationalTestCase to evaluate
        threshold: Minimum required relevancy score (fails if score < threshold).
                   If None, loads from thresholds.yaml (deepeval.multiturn.turn_relevancy_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.turn_relevancy_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Turn relevancy metric score (0-1, higher = more relevant)
            - reason: Explanation of the relevancy assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    return verify_conversational_metric(
        test_case=test_case,
        metric_type='turn_relevancy',
        threshold=threshold,
        model=model,
        include_reason=include_reason,
        case_id=case_id
    )


def evaluate_conversational_test_cases(
    test_cases: List[ConversationalTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[ConversationalTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of conversational test cases using TurnRelevancyMetric.
    
    Args:
        test_cases: List of ConversationalTestCase objects to evaluate
        threshold: Minimum required relevancy score. If None, loads from thresholds.yaml (deepeval.multiturn.turn_relevancy_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.turn_relevancy_model)
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_turn_relevancy_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def verify_role_adherence_metric(
    test_case: ConversationalTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify role adherence using DeepEval RoleAdherenceMetric.
    
    Args:
        test_case: The ConversationalTestCase to evaluate (must have chatbot_role set)
        threshold: Minimum required adherence score (fails if score < threshold).
                   If None, loads from thresholds.yaml (deepeval.multiturn.role_adherence_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.role_adherence_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Role adherence metric score (0-1, higher = better adherence)
            - reason: Explanation of the adherence assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    return verify_conversational_metric(
        test_case=test_case,
        metric_type='role_adherence',
        threshold=threshold,
        model=model,
        include_reason=include_reason,
        case_id=case_id
    )


def evaluate_role_adherence_test_cases(
    test_cases: List[ConversationalTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[ConversationalTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of conversational test cases using RoleAdherenceMetric.
    
    Args:
        test_cases: List of ConversationalTestCase objects to evaluate (must have chatbot_role set)
        threshold: Minimum required adherence score. If None, loads from thresholds.yaml (deepeval.multiturn.role_adherence_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.role_adherence_model)
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_role_adherence_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def verify_knowledge_retention_metric(
    test_case: ConversationalTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
 
    return verify_conversational_metric(
        test_case=test_case,
        metric_type='knowledge_retention',
        threshold=threshold,
        model=model,
        include_reason=include_reason,
        case_id=case_id
    )


def evaluate_knowledge_retention_test_cases(
    test_cases: List[ConversationalTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[ConversationalTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of conversational test cases using KnowledgeRetentionMetric.
    
    Args:
        test_cases: List of ConversationalTestCase objects to evaluate
        threshold: Minimum required retention score. If None, loads from thresholds.yaml (deepeval.multiturn.knowledge_retention_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.knowledge_retention_model)
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_knowledge_retention_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def verify_conversation_completeness_metric(
    test_case: ConversationalTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify conversation completeness using DeepEval ConversationCompletenessMetric.
    
    Args:
        test_case: The ConversationalTestCase to evaluate
        threshold: Minimum required completeness score (fails if score < threshold).
                   If None, loads from thresholds.yaml (deepeval.multiturn.conversation_completeness_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.conversation_completeness_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Conversation completeness metric score (0-1, higher = more complete)
            - reason: Explanation of the completeness assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    return verify_conversational_metric(
        test_case=test_case,
        metric_type='conversation_completeness',
        threshold=threshold,
        model=model,
        include_reason=include_reason,
        case_id=case_id
    )


def evaluate_conversation_completeness_test_cases(
    test_cases: List[ConversationalTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[ConversationalTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of conversational test cases using ConversationCompletenessMetric.
    
    Args:
        test_cases: List of ConversationalTestCase objects to evaluate
        threshold: Minimum required completeness score. If None, loads from thresholds.yaml (deepeval.multiturn.conversation_completeness_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.conversation_completeness_model)
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_conversation_completeness_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def verify_goal_accuracy_metric(
    test_case: ConversationalTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify GoalAccuracyMetric for a conversational test case.
    
    GoalAccuracyMetric evaluates whether the conversation achieved its intended goal.
    This metric can work with tool calls in the conversation turns.
    
    Args:
        test_case: The ConversationalTestCase to evaluate
        threshold: Minimum required goal accuracy score. If None, loads from thresholds.yaml (deepeval.multiturn.goal_accuracy_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.goal_accuracy_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Goal accuracy metric score (0-1, higher = goal achieved better)
            - reason: Explanation of the goal accuracy assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    return verify_conversational_metric(
        test_case=test_case,
        metric_type='goal_accuracy',
        threshold=threshold,
        model=model,
        include_reason=include_reason,
        case_id=case_id
    )


def evaluate_goal_accuracy_test_cases(
    test_cases: List[ConversationalTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[ConversationalTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of conversational test cases using GoalAccuracyMetric.
    
    GoalAccuracyMetric evaluates whether the conversation achieved its intended goal.
    This metric can work with tool calls in the conversation turns.
    
    Args:
        test_cases: List of ConversationalTestCase objects to evaluate
        threshold: Minimum required goal accuracy score. If None, loads from thresholds.yaml (deepeval.multiturn.goal_accuracy_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.goal_accuracy_model)
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_goal_accuracy_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results


def verify_topic_adherence_metric(
    test_case: ConversationalTestCase,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify TopicAdherenceMetric for a conversational test case.
    
    TopicAdherenceMetric evaluates whether the conversation stays on topic and adheres to the intended topic.
    This metric can work with tool calls in the conversation turns.
    
    Args:
        test_case: The ConversationalTestCase to evaluate
        threshold: Minimum required topic adherence score. If None, loads from thresholds.yaml (deepeval.multiturn.topic_adherence_metric_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.topic_adherence_metric_model)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Topic adherence metric score (0-1, higher = better topic adherence)
            - reason: Explanation of the topic adherence assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    print(f"🔍 Debug: Topics: {test_case.topics}")
    # Debug: Verify which metric is being used
    if case_id:
        print(f"🔍 Debug: Evaluating TopicAdherenceMetric for {case_id}")
    
    try:
        # Validate that topics are present
        topics = getattr(test_case, 'topics', None)
        if not topics:
            # Try to get topics from thresholds.yaml
            topics = _get_topic_adherence_topics()
        
        # Ensure topics is a list and not empty
        if not topics:
            error_msg = f"ConversationalTestCase must have topics set for TopicAdherenceMetric"
            if case_id:
                error_msg += f" (case {case_id})"
            return None, None, False, error_msg
        
        # Ensure topics is a list type
        if not isinstance(topics, list):
            # Convert to list if it's not already
            topics = list(topics) if topics else []
        
        if len(topics) == 0:
            error_msg = f"ConversationalTestCase must have a non-empty list of topics for TopicAdherenceMetric"
            if case_id:
                error_msg += f" (case {case_id})"
            return None, None, False, error_msg
        
        print(f"🔍 Debug: Topics type: {type(topics)}, value: {topics}, length: {len(topics)}")
        
        # Use config defaults if not provided
        if threshold is None:
            threshold = _get_topic_adherence_threshold()
        if model is None:
            model = _get_topic_adherence_model()
        
        # Create and measure metric with topics
        # Debug: Verify metric class name
        if case_id:
            print(f"🔍 Debug: Instantiating TopicAdherenceMetric with topics (type: {type(topics)}, len: {len(topics)}): {topics}")
        
        # TopicAdherenceMetric requires relevant_topics as the first required parameter
        # Based on signature: (self, relevant_topics: List[str], threshold: float = 0.5, ...)
        # Pass as first positional argument to ensure it's recognized
        try:
            metric = TopicAdherenceMetric(
                topics,  # First positional argument: relevant_topics (List[str])
                threshold=threshold,
                model=model,
                include_reason=include_reason
            )
        except TypeError as e:
            error_msg = f"Error creating TopicAdherenceMetric: {str(e)}. Topics: {topics}, type: {type(topics)}, is_list: {isinstance(topics, list)}"
            if case_id:
                error_msg += f" (case {case_id})"
            return None, None, False, error_msg
        metric.measure(test_case)
        
        score = metric.score
        reason = metric.reason if include_reason else None
        passed = score >= threshold
        
        return score, reason, passed, None
        
    except Exception as e:
        error_msg = f"Error evaluating TopicAdherenceMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def evaluate_topic_adherence_test_cases(
    test_cases: List[ConversationalTestCase],
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    include_reason: bool = True
) -> List[Tuple[ConversationalTestCase, float, Optional[str], bool, Optional[str]]]:
    """
    Evaluate a list of conversational test cases using TopicAdherenceMetric.
    
    TopicAdherenceMetric evaluates whether the conversation stays on topic and adheres to the intended topic.
    This metric can work with tool calls in the conversation turns.
    
    Args:
        test_cases: List of ConversationalTestCase objects to evaluate
        threshold: Minimum required topic adherence score. If None, loads from thresholds.yaml (deepeval.multiturn.topic_adherence_metric_min)
        model: Model to use for evaluation. If None, loads from thresholds.yaml (deepeval.multiturn.topic_adherence_metric_model)
        include_reason: Whether to include reason in metric (default: True)
        
    Returns:
        List of tuples: (test_case, score, reason, passed, error_message)
    """
    results = []
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        score, reason, passed, error_msg = verify_topic_adherence_metric(
            test_case=test_case,
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            case_id=case_id
        )
        results.append((test_case, score, reason, passed, error_msg))
    
    return results

