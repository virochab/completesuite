"""Pytest fixtures for LLM Application testing."""

import json
import os
import sys
import pytest
import yaml
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to Python path to enable imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from app.client import ApplicationClient


@pytest.fixture
def client():
    """
    Unified client fixture for all application tests.
    
    Supports:
    - Basic chat queries (ask method)
    - User data management (request_delete_my_data method)
    - Tool calling (response includes tools_called attribute)
    """
    return ApplicationClient()


@pytest.fixture
def thresholds():
    """Load compliance thresholds from config."""
    thresholds_path = Path(__file__).parent / "config" / "thresholds.yaml"
    with open(thresholds_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def data_config():
    """Load demographic data configuration."""
    config_path = Path(__file__).parent.parent / "testData" / "config" / "data_config.json"
    with open(config_path, "r") as f:
        return json.load(f)


@pytest.fixture
def deepeval_runner():
    """Fixture for DeepEval test runner."""
    # TODO: Implement DeepEval runner initialization
    # from app.eval import DeepEvalRunner
    # return DeepEvalRunner()
    pass

