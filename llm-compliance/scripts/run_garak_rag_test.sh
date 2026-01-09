#!/bin/bash
# Script to run Garak tests against the RAG API endpoint
# This uses the .venv-garak environment to avoid dependency conflicts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_DIR="$(cd "$PROJECT_ROOT/.." && pwd)"
GARAK_VENV="$PARENT_DIR/.venv-garak"
CONFIG_FILE="$PROJECT_ROOT/config/garak_rag_config.json"
API_URL="${RAG_API_URL:-http://localhost:8000}"

if [ ! -d "$GARAK_VENV" ]; then
    echo "Error: .venv-garak not found at $GARAK_VENV"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Garak config file not found at $CONFIG_FILE"
    exit 1
fi

echo "Using Garak virtual environment: $GARAK_VENV"
echo "Testing RAG API at: $API_URL/query"
echo "Using config: $CONFIG_FILE"

# Update the URI in config if API_URL is set
if [ "$API_URL" != "http://localhost:8000" ]; then
    sed -i.bak "s|http://localhost:8000|$API_URL|g" "$CONFIG_FILE"
fi

# Run garak with REST endpoint (using new command format)
# URI is specified in the JSON config file using -G option
"$GARAK_VENV/bin/garak" --target_type rest -G "$CONFIG_FILE"

# Restore original config if it was modified
if [ -f "$CONFIG_FILE.bak" ]; then
    mv "$CONFIG_FILE.bak" "$CONFIG_FILE"
fi

