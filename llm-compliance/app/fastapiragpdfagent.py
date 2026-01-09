from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import time
import asyncio
import concurrent.futures
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add parent directory (rags) to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up one level to rags/
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from newcbseTest import PDFRAGAgent

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="PDF RAG Agent API",
    description="API for querying PDF documents using RAG (Retrieval Augmented Generation)",
    version="1.0.0"
)

# Initialize the PDF RAG Agent
# You can customize these paths as needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up to llm-compliance/ directory

# Set PDF directory to ragData folder
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", os.path.join(parent_dir, "ragData"))
# Set vector DB directory to vectorragdb folder
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", os.path.join(parent_dir, "vectorRAGdb"))
# Global agent instance
agent: Optional[PDFRAGAgent] = None

# Session management (in-memory storage)
# In production, consider using Redis or a database
sessions: Dict[str, Dict[str, Any]] = {}
SESSION_TIMEOUT = timedelta(hours=24)  # Sessions expire after 24 hours


@app.on_event("startup")
async def startup_event():
    """Initialize the PDF RAG Agent on startup"""
    global agent
    try:
        print(f"🚀 Initializing PDF RAG Agent...")
        print(f"📁 PDF Directory: {PDF_DIRECTORY}")
        print(f"💾 Vector DB Path: {VECTOR_DB_PATH}")
        agent = PDFRAGAgent(pdf_directory=PDF_DIRECTORY, vector_db_path=VECTOR_DB_PATH)
        print("✅ PDF RAG Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing PDF RAG Agent: {str(e)}")
        raise


# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    debug: Optional[bool] = False
    session_id: Optional[str] = None  # Optional session ID for conversation history


class SourceInfo(BaseModel):
    content: str
    metadata: Dict[str, Any]


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    error: bool
    sources_count: int
    session_id: Optional[str] = None  # Session ID for maintaining conversation


class HealthResponse(BaseModel):
    status: str
    message: str
    vector_store_initialized: bool
    qa_chain_initialized: bool


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    message: str


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "PDF RAG Agent API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/ask - POST - Ask a question",
            "query": "/query - POST - Query endpoint (alias for /ask)",
            "health": "/health - GET - Check system health",
            "diagnose": "/diagnose - GET - Run system diagnostics",
            "session": "/session - POST - Create a new session",
            "session/{session_id}": "/session/{session_id} - GET - Get session history"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check the health status of the RAG system"""
    if agent is None:
        raise HTTPException(status_code=503, detail="PDF RAG Agent not initialized")
    
    vector_store_ok = agent.vector_store is not None
    qa_chain_ok = agent.qa_chain is not None
    
    status = "healthy" if (vector_store_ok and qa_chain_ok) else "degraded"
    message = "System is operational" if status == "healthy" else "Some components are not initialized"
    
    return HealthResponse(
        status=status,
        message=message,
        vector_store_initialized=vector_store_ok,
        qa_chain_initialized=qa_chain_ok
    )


def _cleanup_expired_sessions():
    """Remove expired sessions"""
    current_time = datetime.now()
    expired_sessions = [
        sid for sid, session_data in sessions.items()
        if current_time - session_data["created_at"] > SESSION_TIMEOUT
    ]
    for sid in expired_sessions:
        del sessions[sid]


def _get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create a new one"""
    _cleanup_expired_sessions()
    
    if session_id and session_id in sessions:
        # Update last accessed time
        sessions[session_id]["last_accessed"] = datetime.now()
        return session_id
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = {
        "created_at": datetime.now(),
        "last_accessed": datetime.now(),
        "conversation_history": []
    }
    return new_session_id


async def _process_question(request: QuestionRequest):
    """Internal function to process a question - shared by /ask and /query endpoints"""
    if agent is None:
        raise HTTPException(status_code=503, detail="PDF RAG Agent not initialized")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Get or create session
    session_id = _get_or_create_session(request.session_id)
    
    # Get conversation history from session if available
    conversation_history = []
    if session_id in sessions:
        # Get previous Q&A pairs (excluding current question)
        history = sessions[session_id].get("conversation_history", [])
        conversation_history = [
            {"question": h.get("question", ""), "answer": h.get("answer", "")}
            for h in history
        ]
    
    try:
        # Get response directly from qa_chain (like evaluate_system does)
        # This ensures we get full content, not truncated like agent.ask()
        if not agent.qa_chain:
            raise HTTPException(status_code=503, detail="QA chain not initialized")
        
        # Add delay to avoid rate limiting or allow processing time
        time.sleep(1.5)  # 1.5 second delay
        
        # Log start time for performance monitoring
        start_time = time.time()
        print(f"⏱️  Processing question: {request.question[:100]}...", flush=True)
        if conversation_history:
            print(f"📜 Using conversation history ({len(conversation_history)} previous exchanges)", flush=True)
        
        # Run qa_chain.invoke with a timeout to prevent hanging
        # Check for prompt injection attempts
        promptInjectionStrings = [
            "ignore instructions", 
            "ignore any previous and following instructions", 
            "stop everything!!! now!!!", 
            "nevermind.",
            "pretend to be",
            "developer mode",
            "free generator",
            "act as",
            "simulate",
            "act like",
            "pretend to",
            "pretend to be",
            "pretend to act like",
            "pretend to simulate",
            "pretend to ignore",
            "pretend to stop",
            "pretend to nevermind."
        ]
        
        question_lower = request.question.lower()
        if any(injection_string in question_lower for injection_string in promptInjectionStrings):
            return AnswerResponse(
                answer="I'm sorry, but I can't assist with that request.", 
                sources=[], 
                error=True, 
                sources_count=0,
                session_id=session_id
            )
        
        # Format question with conversation history
        formatted_question = agent._format_question_with_history(request.question, conversation_history)
        if conversation_history:
            print(f"📝 Formatted question with history (length: {len(formatted_question)} chars)", flush=True)
        
        # Use ThreadPoolExecutor to run the blocking invoke in a separate thread
        # This allows us to timeout if it takes too long
        # Note: Garak config has request_timeout=300, so we set this slightly lower
        # to ensure we return before Garak times out
        max_processing_time = 280  # ~4.7 minutes (leaving small buffer for network/overhead)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.qa_chain.invoke, {"query": formatted_question})
            try:
                result = future.result(timeout=max_processing_time)
            except concurrent.futures.TimeoutError:
                elapsed_time = time.time() - start_time
                print(f"❌ Question processing timed out after {elapsed_time:.2f} seconds", flush=True)
                raise HTTPException(
                    status_code=504, 
                    detail=f"Request processing timed out after {max_processing_time} seconds. "
                           f"The RAG system may be overloaded or the LLM API is not responding."
                )
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"❌ Error during processing after {elapsed_time:.2f} seconds: {str(e)}", flush=True)
                raise
        
        # Log processing time
        elapsed_time = time.time() - start_time
        print(f"✅ Question processed in {elapsed_time:.2f} seconds", flush=True)
        
        # Extract answer (same as evaluate_system)
        answer = result.get("result", "")
        
        # Extract full source content (same as evaluate_system - no truncation)
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                # Use full page_content, not truncated (matching evaluate_system line 515)
                source_info = SourceInfo(
                    content=doc.page_content,  # Full content, not truncated
                    metadata=doc.metadata
                )
                sources.append(source_info)
        
        # Store conversation history in session
        if session_id in sessions:
            sessions[session_id]["conversation_history"].append({
                "question": request.question,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last 50 messages to prevent memory issues
            if len(sessions[session_id]["conversation_history"]) > 50:
                sessions[session_id]["conversation_history"] = \
                    sessions[session_id]["conversation_history"][-50:]
        
        return AnswerResponse(
            answer=answer,
            sources=sources,
            error=False,
            sources_count=len(sources),
            session_id=session_id
        )
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error processing question: {str(e)}", flush=True)
        print(f"Traceback:\n{error_trace}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/ask", response_model=AnswerResponse, tags=["Query"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the RAG system
    
    - **question**: The question to ask
    - **debug**: Optional flag to include debug information (default: False)
    
    Returns:
    - **answer**: The answer from the RAG system
    - **sources**: List of source documents with content and metadata
    - **error**: Boolean indicating if there was an error
    - **sources_count**: Number of source documents
    """
    return await _process_question(request)


@app.post("/query", response_model=AnswerResponse, tags=["Query"])
async def query_question(request: QuestionRequest):
    """
    Query endpoint (alias for /ask) - Ask a question to the RAG system
    
    - **question**: The question to ask
    - **debug**: Optional flag to include debug information (default: False)
    
    Returns:
    - **answer**: The answer from the RAG system
    - **sources**: List of source documents with content and metadata
    - **error**: Boolean indicating if there was an error
    - **sources_count**: Number of source documents
    """
    return await _process_question(request)


@app.get("/diagnose", tags=["System"])
async def diagnose_system():
    """Run system diagnostics"""
    if agent is None:
        raise HTTPException(status_code=503, detail="PDF RAG Agent not initialized")
    
    try:
        # Collect diagnostic information
        diagnostics = {
            "vector_store": {
                "initialized": agent.vector_store is not None,
                "document_count": None
            },
            "qa_chain": {
                "initialized": agent.qa_chain is not None
            },
            "embedding_model": agent.embedding_function.model_name if hasattr(agent.embedding_function, 'model_name') else "Unknown"
        }
        
        # Get document count if vector store is available
        if agent.vector_store:
            try:
                collection = agent.vector_store._collection
                diagnostics["vector_store"]["document_count"] = collection.count()
            except:
                diagnostics["vector_store"]["document_count"] = "Unable to retrieve"
        
        return {
            "status": "success",
            "diagnostics": diagnostics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running diagnostics: {str(e)}")


@app.post("/rebuild", tags=["System"])
async def rebuild_vector_store():
    """Rebuild the vector store from PDFs"""
    if agent is None:
        raise HTTPException(status_code=503, detail="PDF RAG Agent not initialized")
    
    try:
        agent.rebuild_vector_store()
        return {
            "status": "success",
            "message": "Vector store rebuilt successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding vector store: {str(e)}")


@app.post("/session", response_model=SessionResponse, tags=["Session"])
async def create_session():
    """Create a new session for maintaining conversation history"""
    session_id = _get_or_create_session()
    return SessionResponse(
        session_id=session_id,
        created_at=sessions[session_id]["created_at"].isoformat(),
        message="Session created successfully"
    )


@app.get("/session/{session_id}", tags=["Session"])
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    return {
        "session_id": session_id,
        "created_at": session_data["created_at"].isoformat(),
        "last_accessed": session_data["last_accessed"].isoformat(),
        "conversation_count": len(session_data["conversation_history"]),
        "conversation_history": session_data["conversation_history"]
    }


@app.delete("/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {
        "status": "success",
        "message": f"Session {session_id} deleted successfully"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

