"""
Simple script to run the FastAPI RAG server
Run this script to start the server, then use accessFastAPIrag.py to interact with it
"""
import uvicorn
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

if __name__ == "__main__":
    print("=" * 60)
    print("Starting PDF RAG Agent FastAPI Server")
    print("=" * 60)
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Interactive API: http://localhost:8000/redoc")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "fastapiragpdfagent:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

