from datetime import datetime
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from dotenv import load_dotenv
from pathlib import Path
import json
import os
import yaml
load_dotenv()


# Get the project root directory (parent of utils)
project_root = Path(__file__).parent.parent

def load_config():
    """Load configuration from syntheticData_config.yaml"""
    config_path = project_root / "testData" / "config" / "syntheticData_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('syntheticData_config', {})

# Load config
_config = load_config()

def save_goldens_to_json(goldens, directory, filename_prefix="goldens"):
    """
    Save goldens to a JSON file manually.
    
    Args:
        goldens: List of Golden or ConversationalGolden objects
        directory: Directory path where to save the file
        filename_prefix: Prefix for the filename
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Convert goldens to dictionaries
    goldens_data = []
    for golden in goldens:
        golden_dict = {}
        # Handle both regular Golden and ConversationalGolden
        if hasattr(golden, 'input'):
            # Regular Golden
            golden_dict = {
                "input": golden.input,
                "expected_output": golden.expected_output if hasattr(golden, 'expected_output') else None,
                "context": golden.context if hasattr(golden, 'context') else None,
                "actual_output": None,  # To be filled later when testing
            }
        elif hasattr(golden, 'scenario'):
            # ConversationalGolden
            golden_dict = {
                "scenario": golden.scenario,
                "expected_outcome": golden.expected_outcome if hasattr(golden, 'expected_outcome') else None,
                "user_description": golden.user_description if hasattr(golden, 'user_description') else None,
                "context": golden.context if hasattr(golden, 'context') else None,
                "turns": [
                    {
                        "role": getattr(turn, 'role', 'unknown'),
                        "content": getattr(turn, 'content', str(turn))
                    }
                    for turn in (golden.turns if hasattr(golden, 'turns') and golden.turns else [])
                ],
            }
        
        # Add metadata if available
        if hasattr(golden, 'additional_metadata'):
            golden_dict["additional_metadata"] = golden.additional_metadata
        
        goldens_data.append(golden_dict)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = os.path.join(directory, filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(goldens_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(goldens)} goldens to: {filepath}")
    return filepath

def generate_single_turn_goldens():
    """Generate single-turn goldens using config from syntheticData_config.yaml"""
    # Get config values
    singleturn_config = _config.get('singleturn', {})
    goldens_config = singleturn_config.get('goldens', {})
    context_config = _config.get('context_construction_config', {})
    
    # Get document paths from config - ensure it's a list
    document_paths = goldens_config.get('document_paths', ['ragData/eeev102.pdf'])
    # Ensure document_paths is a list (handle case where config might have a single string)
    if isinstance(document_paths, str):
        document_paths = [document_paths]
    elif not isinstance(document_paths, list):
        document_paths = ['ragData/eeev102.pdf']  # Fallback to default
    # Convert relative paths to absolute paths
    document_paths = [str(project_root / path) if not Path(path).is_absolute() else path for path in document_paths]
    
    # Create ContextConstructionConfig from config
    context_construction_config = ContextConstructionConfig(
        critic_model=context_config.get('critic_model', 'gpt-4o'),
        encoding=context_config.get('encoding') if context_config.get('encoding') else None,
        max_contexts_per_document=context_config.get('max_contexts_per_document', 3),
        min_contexts_per_document=context_config.get('min_contexts_per_document', 1),
        max_context_length=context_config.get('max_context_length', 3),
        min_context_length=context_config.get('min_context_length', 1),
        chunk_size=context_config.get('chunk_size', 1024),
        chunk_overlap=context_config.get('chunk_overlap', 0),
        context_quality_threshold=context_config.get('context_quality_threshold', 0.5),
        context_similarity_threshold=context_config.get('context_similarity_threshold', 0.5),
        max_retries=context_config.get('max_retries', 3),
        embedder=context_config.get('embedder', 'text-embedding-3-small')
    )
    
    # Create Synthesizer with context_construction_config
    synthesizer = Synthesizer()
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        include_expected_output=True,
        max_goldens_per_context=goldens_config.get('max_goldens_per_context', 1),
        context_construction_config=context_construction_config
    )
   
    print(f"\nGenerated {len(goldens)} golden test cases from documentation:")
    for i, golden in enumerate(goldens, 1):
        print(f"\n--- Golden Test Case {i} ---")
        print(f"Input: {golden.input}")
        print(f"Expected Output: {golden.expected_output}")
        print(f"Context: {golden.context[:200]}..." if len(golden.context) > 200 else f"Context: {golden.context}")

    # Save to directory from config
    save_dir = project_root / goldens_config.get('directory', 'testData/synthetic_data/singleturn/singleturnGoldens')
    synthesizer.save_as(
        file_type='json',
        directory=str(save_dir)
    )
    return goldens

def generate_multiturn_goldens():
    """Generate multi-turn conversational goldens using config from syntheticData_config.yaml"""
    # Get config values
    multiturn_config = _config.get('multiturn', {})
    goldens_config = multiturn_config.get('goldens', {})
    context_config = _config.get('context_construction_config', {})
    
    # Get document paths from config - ensure it's a list
    document_paths = goldens_config.get('document_paths', ['ragData/eeev102.pdf'])
    # Ensure document_paths is a list (handle case where config might have a single string)
    if isinstance(document_paths, str):
        document_paths = [document_paths]
    elif not isinstance(document_paths, list):
        document_paths = ['ragData/eeev102.pdf']  # Fallback to default
    # Convert relative paths to absolute paths
    document_paths = [str(project_root / path) if not Path(path).is_absolute() else path for path in document_paths]
    
    # Create ContextConstructionConfig from config
    context_construction_config = ContextConstructionConfig(
        critic_model=context_config.get('critic_model', 'gpt-4o'),
        encoding=context_config.get('encoding') if context_config.get('encoding') else None,
        max_contexts_per_document=context_config.get('max_contexts_per_document', 3),
        min_contexts_per_document=context_config.get('min_contexts_per_document', 1),
        max_context_length=context_config.get('max_context_length', 3),
        min_context_length=context_config.get('min_context_length', 1),
        chunk_size=context_config.get('chunk_size', 1024),
        chunk_overlap=context_config.get('chunk_overlap', 0),
        context_quality_threshold=context_config.get('context_quality_threshold', 0.5),
        context_similarity_threshold=context_config.get('context_similarity_threshold', 0.5),
        max_retries=context_config.get('max_retries', 3),
        embedder=context_config.get('embedder', 'text-embedding-3-small')
    )
    
    # Create Synthesizer with context_construction_config
    synthesizer = Synthesizer()
    # For conversational goldens, use 'include_expected_outcome' instead of 'include_expected_output'
    # Reference: https://deepeval.com/docs/synthesizer-introduction
    # Note: user_description is not a valid parameter for generate_conversational_goldens_from_docs()
    conversational_goldens = synthesizer.generate_conversational_goldens_from_docs(
        document_paths=document_paths,
        include_expected_outcome=goldens_config.get('include_expected_outcome', True),
        max_goldens_per_context=goldens_config.get('max_goldens_per_context', 2),
        context_construction_config=context_construction_config
    )
    
    # Set user_description from config
    user_description = goldens_config.get('user_description')
    if user_description:
        for golden in conversational_goldens:
            golden.user_description = user_description
    
    print(conversational_goldens)
    print(f"\nGenerated {len(conversational_goldens)} conversational golden test cases from documentation:")
    for i, golden in enumerate(conversational_goldens, 1):
        print(f"\n--- Conversational Golden Test Case {i} ---")
        # ConversationalGolden has different attributes than regular Golden
        # Reference: https://deepeval.com/docs/synthesizer-introduction
        if hasattr(golden, 'scenario'):
            print(f"Scenario: {golden.scenario}")
        if hasattr(golden, 'user_description'):
            print(f"User Description: {golden.user_description}")
        if hasattr(golden, 'expected_outcome'):
            print(f"Expected Outcome: {golden.expected_outcome}")
        if hasattr(golden, 'turns') and golden.turns:
            print(f"Turns ({len(golden.turns)}):")
            for j, turn in enumerate(golden.turns, 1):
                role = getattr(turn, 'role', 'unknown')
                content = getattr(turn, 'content', str(turn))
                print(f"  Turn {j} ({role}): {content[:100]}..." if len(content) > 100 else f"  Turn {j} ({role}): {content}")
        if hasattr(golden, 'context'):
            context_str = str(golden.context) if not isinstance(golden.context, list) else '\n'.join(golden.context)
            print(f"Context: {context_str[:200]}..." if len(context_str) > 200 else f"Context: {context_str}")

    # Save goldens manually since synthesizer.save_as() doesn't have access to returned goldens
    # Get directory and file prefix from config
    save_dir = project_root / goldens_config.get('directoryPrefix', goldens_config.get('directory', 'testData/synthetic_data/multiturn/conversationalGoldens'))
    file_prefix = goldens_config.get('filePrefx', goldens_config.get('filePrefix', 'conversationalGoldens_'))
    save_goldens_to_json(conversational_goldens, str(save_dir), file_prefix.rstrip('_'))
    return conversational_goldens

if __name__ == "__main__":
    generate_multiturn_goldens()