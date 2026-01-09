"""RAG PDF client for LLM testing with source citations and context verification."""
import os
import sys
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up to llm-compliance/
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import PDFRAGAgent - try app.newcbseTest first (when imported from parent), 
# then fallback to direct import (when run from app directory)
try:
    from app.newcbseTest import PDFRAGAgent
except ImportError:
    try:
        from newcbseTest import PDFRAGAgent
    except ImportError:
        # Last resort: add current directory and try again
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from newcbseTest import PDFRAGAgent


class RAGResponse:
    """Response object for RAG queries with citations and source files."""
    
    def __init__(self, text: str, citations: List[Dict[str, Any]] = None, 
                 tools_called: List[str] = None, context: List[str] = None,
                 source_files: List[str] = None, metadata: Dict[str, Any] = None):
        """
        Initialize RAG Response object.
        
        Args:
            text: The answer text from the RAG system
            citations: List of citation dictionaries with content and metadata
            tools_called: List of tools/capabilities used (e.g., ['retrieval', 'llm'])
            context: List of context chunks used for answering
            source_files: List of source file paths/names
            metadata: Additional metadata about the response
        """
        self.text = text
        self.citations = citations or []
        self.tools_called = tools_called or []
        self.context = context or []
        self.source_files = source_files or []
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"RAGResponse(text={self.text[:50]}..., citations={len(self.citations)}, source_files={len(self.source_files)})"


class RAGClient:
    """Client for RAG PDF Agent testing with source citations and verification."""
    
    def __init__(self, pdf_directory: Optional[str] = None, 
                 vector_db_path: Optional[str] = None,
                 api_url: Optional[str] = None,
                 use_api: bool = False,
                 api_endpoint: str = "/ask"):
        """
        Initialize the RAG Client.
        
        Args:
            pdf_directory: Directory containing PDF files (defaults to ragData folder)
            vector_db_path: Path to vector database storage (defaults to vectorRAGdb folder)
            api_url: URL of the FastAPI RAG service (e.g., "http://localhost:8000")
            use_api: If True, use FastAPI service; if False, use direct PDFRAGAgent
            api_endpoint: API endpoint path (default: "/ask", can be "/query" or custom)
        """
        self.use_api = use_api
        self.api_url = api_url or os.getenv("RAG_API_URL", "http://localhost:8000")
        self.api_endpoint = api_endpoint
        
        if not use_api:
            # Initialize PDFRAGAgent directly
            self.agent = PDFRAGAgent(pdf_directory=pdf_directory, vector_db_path=vector_db_path)
        else:
            self.agent = None
            # Test API connection
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"✅ Connected to RAG API at {self.api_url}")
                else:
                    print(f"⚠️  API health check returned status {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Could not connect to RAG API at {self.api_url}: {e}")
                print("💡 Falling back to direct agent mode...")
                self.use_api = False
                self.agent = PDFRAGAgent(pdf_directory=pdf_directory, vector_db_path=vector_db_path)
    
    def ask(self, query: str, debug: bool = False) -> RAGResponse:
        """
        Send a query to the RAG system and return response with citations.
        
        Args:
            query: The question to ask
            debug: If True, include debug information
            
        Returns:
            RAGResponse object with text, citations, context, and source files
        """
        if self.use_api:
            return self._ask_via_api(query, debug)
        else:
            return self._ask_direct(query, debug)
    
    def _ask_via_api(self, query: str, debug: bool = False) -> RAGResponse:
        """Query via FastAPI service."""
        try:
            response = requests.post(
                f"{self.api_url}{self.api_endpoint}",
                json={"question": query, "debug": debug},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract source files from metadata
            source_files = []
            citations = []
            context_chunks = []
            
            for source in data.get("sources", []):
                # Extract source file from metadata
                metadata = source.get("metadata", {})
                source_file = metadata.get("source", "")
                if source_file:
                    # Extract just the filename
                    source_file_name = os.path.basename(source_file)
                    if source_file_name not in source_files:
                        source_files.append(source_file_name)
                
                # Build citation
                citation = {
                    "content": source.get("content", ""),
                    "metadata": metadata,
                    "source_file": source_file_name if source_file else "Unknown"
                }
                citations.append(citation)
                context_chunks.append(source.get("content", ""))
            
            return RAGResponse(
                text=data.get("answer", ""),
                citations=citations,
                tools_called=["retrieval", "llm"] if data.get("sources_count", 0) > 0 else ["llm"],
                context=context_chunks,
                source_files=source_files,
                metadata={
                    "sources_count": data.get("sources_count", 0),
                    "error": data.get("error", False)
                }
            )
            
        except requests.exceptions.RequestException as e:
            return RAGResponse(
                text=f"Error connecting to RAG API: {str(e)}",
                citations=[],
                tools_called=[],
                metadata={"error": True, "error_type": "api_connection"}
            )
    
    def _ask_direct(self, query: str, debug: bool = False) -> RAGResponse:
        """Query directly using PDFRAGAgent."""
        if not self.agent:
            return RAGResponse(
                text="RAG Agent not initialized",
                citations=[],
                tools_called=[],
                metadata={"error": True}
            )
        
        try:
            # Use qa_chain directly to get full source documents (not truncated)
            if not self.agent.qa_chain:
                return RAGResponse(
                    text="RAG system not initialized. Please ensure PDF files are available.",
                    citations=[],
                    tools_called=[],
                    metadata={"error": True}
                )
            
            result = self.agent.qa_chain.invoke({"query": query})
            
            # Extract answer
            answer = result.get("result", "")
            
            # Extract source information with full content
            citations = []
            context_chunks = []
            source_files = []
            
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    # Extract source file from metadata
                    metadata = doc.metadata
                    source_file = metadata.get("source", "")
                    source_file_name = os.path.basename(source_file) if source_file else "Unknown"
                    
                    if source_file_name not in source_files and source_file_name != "Unknown":
                        source_files.append(source_file_name)
                    
                    # Build citation with full content
                    citation = {
                        "content": doc.page_content,  # Full content, not truncated
                        "metadata": metadata,
                        "source_file": source_file_name,
                        "page": metadata.get("page", None)
                    }
                    citations.append(citation)
                    context_chunks.append(doc.page_content)
            
            return RAGResponse(
                text=answer,
                citations=citations,
                tools_called=["retrieval", "llm"] if citations else ["llm"],
                context=context_chunks,
                source_files=source_files,
                metadata={
                    "sources_count": len(citations),
                    "error": False
                }
            )
            
        except Exception as e:
            return RAGResponse(
                text=f"Error processing question: {str(e)}",
                citations=[],
                tools_called=[],
                metadata={"error": True, "error_message": str(e)}
            )
    
    def get_source_files(self) -> List[str]:
        """Get list of all source PDF files in the directory."""
        if self.use_api:
            # Try to get from API if available
            try:
                response = requests.get(f"{self.api_url}/diagnose", timeout=10)
                if response.status_code == 200:
                    # API doesn't return file list, so we can't get it
                    return []
            except:
                pass
            return []
        else:
            if not self.agent:
                return []
            # Get PDF files from the directory
            import glob
            pdf_files = glob.glob(os.path.join(self.agent.pdf_directory, "*.pdf"))
            return [os.path.basename(f) for f in pdf_files]
    
    def diagnose(self):
        """Run system diagnostics."""
        if self.use_api:
            try:
                response = requests.get(f"{self.api_url}/diagnose", timeout=10)
                if response.status_code == 200:
                    print(response.json())
                else:
                    print(f"API returned status {response.status_code}")
            except Exception as e:
                print(f"Error connecting to API: {e}")
        else:
            if self.agent:
                self.agent.diagnose_system()
            else:
                print("Agent not initialized")


# Example usage
if __name__ == "__main__":
    # Initialize client (direct mode)
    client = RAGClient(use_api=False)
    
    # Or use API mode
    # client = RAGClient(use_api=True, api_url="http://localhost:8000")
    
    # Ask a question
    query = "What is the Namami Gange programme?"
    response = client.ask(query)
    
    print(f"\n📝 Answer: {response.text}")
    print(f"\n📚 Source Files: {response.source_files}")
    print(f"\n🔗 Citations ({len(response.citations)}):")
    for i, citation in enumerate(response.citations, 1):
        print(f"\n  Citation {i}:")
        print(f"    Source File: {citation.get('source_file', 'Unknown')}")
        print(f"    Page: {citation.get('page', 'N/A')}")
        print(f"    Content Preview: {citation.get('content', '')[:100]}...")
    
    print(f"\n🛠️  Tools Used: {response.tools_called}")
    print(f"\n📊 Metadata: {response.metadata}")

