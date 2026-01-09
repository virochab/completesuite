from datetime import date
import os
from langchain_core.documents import Document
import openai
import argparse
import yaml
from pathlib import Path
# Remove RAGAS imports to avoid the conflict - we'll add them back conditionally
from dotenv import load_dotenv
from datasets import Dataset 
from ragas import evaluate
from langchain_openai import ChatOpenAI
from langchain_classic.chains  import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
from ragas.embeddings import LangchainEmbeddingsWrapper
import glob
from typing import List, Dict, Any, Optional

class PDFRAGAgent:
    def __init__(self, pdf_directory: str = None, vector_db_path: str = None):
        """
        Initialize the PDF RAG Agent
        
        Args:
            pdf_directory: Directory containing PDF files (defaults to ragData folder)
            vector_db_path: Path to store/load vector database (defaults to vectorragdb folder)
        """
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_key = api_key
        
        # Default to ragData directory if not specified
   
        
        # Default to vectorragdb directory if not specified
        if vector_db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)  # Go up to llm-compliance/ directory
            vector_db_path = os.path.join(parent_dir, "vectorInsurancedb")
        
        self.pdf_directory = pdf_directory
        self.vector_db_path = vector_db_path
        # Better embedding model for improved retrieval quality
        # Options: "all-MiniLM-L6-v2" (fast, lower quality), 
        #          "all-mpnet-base-v2" (better quality, recommended),
        #          "BAAI/bge-small-en-v1.5" (excellent for retrieval)
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
        
        # Validate and fix common incorrect model names
        invalid_models = {
            "sentence-transformers/openai": "all-mpnet-base-v2",
            "openai": "all-mpnet-base-v2",
            "sentence-transformers": "all-mpnet-base-v2"
        }
        if embedding_model in invalid_models:
            print(f"⚠️  Warning: Invalid embedding model '{embedding_model}' detected.")
            print(f"   Using default model '{invalid_models[embedding_model]}' instead.")
            embedding_model = invalid_models[embedding_model]
        
        try:
            self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)
            print(f"✅ Using embedding model: {embedding_model}")
        except Exception as e:
            print(f"❌ Error loading embedding model '{embedding_model}': {e}")
            print(f"   Falling back to default: 'all-mpnet-base-v2'")
            self.embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        
        # Use RecursiveCharacterTextSplitter instead of SemanticChunker for better compatibility
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vector_store = None
        self.qa_chain = None
        
        # Initialize Address Verification Tool
        self.address_verifier = AddressVerificationTool()
        self.address_verification_tool = self._create_address_verification_tool()
        print("✅ Address Verification Tool initialized")
        
        self.setup_rag_system()
    
    def load_pdfs(self) -> List[Any]:
        """Load all PDF files from the specified directory"""
        print(f"Loading PDFs from {self.pdf_directory}...")
        
        # Check if directory exists and has PDF files
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_directory}")
            print("Please add PDF files to the directory and restart the agent.")
            return []
        
        print(f"Found {len(pdf_files)} PDF files: {[os.path.basename(f) for f in pdf_files]}")
        
        # Load PDFs using DirectoryLoader for batch processing
        loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        documents = loader.load()
        print(f"Loaded {len(documents)} document pages")
        
        return documents
    
    def process_documents(self, documents: List[Any]) -> List[Any]:
        """Process and clean document content"""
        print("Processing and cleaning documents...")
        
        processed_docs = []
        for doc in documents:
            # Clean the text content
            cleaned_content = (
                str(doc.page_content)
                .replace("\\n", " ")
                .replace("\\t", " ")
                .replace("\n", " ")
                .replace("\t", " ")
                .strip()
            )
            
            # Only add non-empty documents with substantial content
            if cleaned_content and len(cleaned_content) > 50:
                doc.page_content = cleaned_content
                processed_docs.append(doc)
        
        print(f"Processed {len(processed_docs)} document chunks")
        return processed_docs
    
    def create_vector_store(self, documents: List[Any]):
        """Create or load vector store from documents"""
        if os.path.exists(self.vector_db_path) and os.listdir(self.vector_db_path):
            print("Loading existing vector database...")
            try:
                self.vector_store = Chroma(
                    persist_directory=self.vector_db_path,
                    embedding_function=self.embedding_function
                )
                # Test if the vector store is working
                test_results = self.vector_store.similarity_search("test", k=1)
                print(f"Vector database loaded successfully with {len(test_results)} test results")
            except Exception as e:
                print(f"Error loading existing vector database: {e}")
                print("Creating new vector database...")
                self._create_new_vector_store(documents)
        else:
            print("Creating new vector database...")
            self._create_new_vector_store(documents)
    
    def _create_new_vector_store(self, documents: List[Any]):
        """Create a new vector store"""
        if not documents:
            print("No documents to process!")
            return
            
        # Split documents into chunks
        docs = self.text_splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunks")
        
        if not docs:
            print("No chunks created from documents!")
            return
        
        # Create vector store
        try:
            self.vector_store = Chroma.from_documents(
                docs,
                self.embedding_function,
                persist_directory=self.vector_db_path
            )
            print("Vector database created and persisted")
        except Exception as e:
            print(f"Error creating vector store: {e}")
    
    def setup_qa_chain(self, prompt_config_path: Optional[str] = None):
        """
        Setup the question-answering chain
        
        Args:
            prompt_config_path: Optional path to YAML file containing prompt template.
                               Defaults to strict_rag_system.yaml in ragSysPrompts directory.
        """
        if not self.vector_store:
            print("Cannot setup QA chain - vector store not initialized")
            return
        
        # Load prompt template from YAML file
        if prompt_config_path is None:
            # Default to strict RAG system prompt
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_config_path = os.path.join(current_dir, "ragSysPrompts", "lenient_twiarag_system.yaml")
        
        try:
            with open(prompt_config_path, 'r', encoding='utf-8') as f:
                prompt_config = yaml.safe_load(f)
            
            PROMPT_TEMPLATE = prompt_config.get('prompt_template', '')
            if not PROMPT_TEMPLATE:
                raise ValueError(f"Prompt template not found in {prompt_config_path}")
            
            print(f"✅ Loaded prompt template from: {prompt_config_path}")
        except FileNotFoundError:
            print(f"⚠️  Warning: Prompt config file not found at {prompt_config_path}")
            print("   Using default prompt template...")
            # Fallback to default prompt
            PROMPT_TEMPLATE = """
        You are a helpful AI assistant for TWIA (Texas Windstorm Insurance Association) insurance questions. 
        Answer the question using ONLY the information provided in the context below.
        
        You have access to a tool called "verify_address" that can verify if an address is eligible for TWIA coverage.
        Use this tool when users ask about address eligibility, coverage areas, or need to verify if their property 
        location qualifies for TWIA insurance. The tool requires an address, zipcode, and optionally a county name.
        
        IMPORTANT RULES:
        1. Base your answer STRICTLY on the provided context
        2. If the context doesn't contain information to answer the question, respond with: "I cannot find information about this question in the provided documents."
        3. Do not use your general knowledge - only use the context provided
        4. Quote relevant parts from the context when possible
        5. Be specific and cite which document or section the information comes from when available
        6. If a user asks about address eligibility or coverage areas, use the verify_address tool to check their address
        
        Context from documents:
        {context}
        
        Question: {question}
        
        Answer based strictly on the context above. If the question involves address verification, use the verify_address tool:
        """
        except Exception as e:
            print(f"⚠️  Error loading prompt config: {e}")
            print("   Using default prompt template...")
            # Fallback to default prompt
            PROMPT_TEMPLATE = """
        You are a helpful AI assistant for TWIA (Texas Windstorm Insurance Association) insurance questions. 
        Answer the question using ONLY the information provided in the context below.
        
        You have access to a tool called "verify_address" that can verify if an address is eligible for TWIA coverage.
        Use this tool when users ask about address eligibility, coverage areas, or need to verify if their property 
        location qualifies for TWIA insurance. The tool requires an address, zipcode, and optionally a county name.
        
        IMPORTANT RULES:
        1. Base your answer STRICTLY on the provided context
        2. If the context doesn't contain information to answer the question, respond with: "I cannot find information about this question in the provided documents."
        3. Do not use your general knowledge - only use the context provided
        4. Quote relevant parts from the context when possible
        5. Be specific and cite which document or section the information comes from when available
        6. If a user asks about address eligibility or coverage areas, use the verify_address tool to check their address
        
        Context from documents:
        {context}
        
        Question: {question}
        
        Answer based strictly on the context above. If the question involves address verification, use the verify_address tool:
        """
        
        try:
            # Create LLM - Note: RetrievalQA may not handle tool-bound LLMs properly
            # We'll use the LLM without tools for now, and handle tool calls separately if needed
            llm = ChatOpenAI(
                temperature=0, 
                model="gpt-4.1-nano",
                max_tokens=500
            )
            
            # TODO: Tool calling with RetrievalQA needs special handling
            # For now, we'll use the LLM without bound tools to ensure answers work
            # The tool can be called manually when needed for address verification
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,  # Using LLM without bound tools for now
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        'k': 3  # Reduced from 5 to 3: Retrieve top 3 most relevant chunks (reduces input token usage by 20-30%)
                        #'fetch_k': 20  # Fetch more candidates before filtering
                    }
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)}
            )
            print("QA chain setup completed")
            print("⚠️  Note: Address verification tool is available but not auto-called. Use manual verification when needed.")
        except Exception as e:
            print(f"Error setting up QA chain: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_rag_system(self):
        """Initialize the complete RAG system"""
        # Check if vector database already exists - if so, skip loading PDFs
        if os.path.exists(self.vector_db_path) and os.listdir(self.vector_db_path):
            print(f"📁 Existing vector database found at {self.vector_db_path}")
            print("⏭️  Skipping PDF loading and vector store creation...")
            try:
                # Load existing vector store
                self.vector_store = Chroma(
                    persist_directory=self.vector_db_path,
                    embedding_function=self.embedding_function
                )
                # Test if the vector store is working
                test_results = self.vector_store.similarity_search("test", k=1)
                print(f"✅ Vector database loaded successfully")
                # Setup QA chain with existing vector store
                self.setup_qa_chain()
                if self.qa_chain:
                    print("✅ RAG system initialized successfully from existing database!")
                    return  # Exit early - skip all PDF loading and processing below
                else:
                    print("⚠️  QA chain setup failed, will rebuild...")
            except Exception as e:
                print(f"⚠️  Error loading existing vector database: {e}")
                print("Will create new vector database...")
        
        # If we get here, we need to create/rebuild the vector store
        # This code will NOT run if database exists and loads successfully (due to return above)
        print("📚 Loading TWIA sample documents and creating vector database...")
        documents = create_sample_documents()
        
        if not documents:
            print("No documents loaded. RAG system not initialized.")
            return
        
        # Sample documents are already Document objects, but we can still process them for consistency
        # (process_documents will clean and validate the content)
        processed_docs = self.process_documents(documents)
        self.create_vector_store(processed_docs)
        self.setup_qa_chain()
        
        if self.qa_chain:
            print("✅ RAG system initialized successfully!")
        else:
            print("❌ RAG system initialization failed!")
    
    def ask(self, question: str, debug: bool = False, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Ask a question to the RAG system
        
        Args:
            question: The question to ask
            debug: If True, print debug information
            conversation_history: Optional list of previous Q&A pairs in format [{"question": "...", "answer": "..."}]
            
        Returns:
            Dictionary containing answer and source information
        """
        if not self.qa_chain:
            return {
                "answer": "RAG system not initialized. Please ensure PDF files are available and try rebuilding the system.",
                "sources": [],
                "error": True
            }
        
        try:
            # Format question with conversation history context if provided
            formatted_question = self._format_question_with_history(question, conversation_history)
            
            # Debug: Test retrieval directly
            if debug:
                print(f"\n🔍 DEBUG: Testing retrieval for question: '{formatted_question}'")
                retrieved_docs = self.vector_store.similarity_search(formatted_question, k=2)  # Match QA chain retrieval count
                print(f"📄 Retrieved {len(retrieved_docs)} documents:")
                for i, doc in enumerate(retrieved_docs):
                    print(f"  Doc {i+1}: {doc.page_content[:100]}...")
                    print(f"  Metadata: {doc.metadata}")
                print("-" * 50)
            
            result = self.qa_chain.invoke({"query": formatted_question})
            
            # Debug: Print full result structure
            if debug:
                print(f"🔍 Full result structure: {type(result)}")
                print(f"🔍 Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                print(f"🔍 Result content: {result}")
            
            # Extract answer - handle both regular responses and tool call responses
            answer = result.get("result", "")
            
            # Check if result is empty or if it's a tool call response
            if not answer or answer == "":
                # The LLM might have made a tool call instead of answering directly
                # Check if we can get the raw response
                print(f"⚠️  Empty answer detected. Full result type: {type(result)}")
                print(f"⚠️  Full result: {result}")
                
                # Try to get answer from different possible result structures
                if isinstance(result, dict):
                    # Check for different possible keys
                    for key in ["answer", "output", "text", "response"]:
                        if key in result and result[key]:
                            answer = str(result[key])
                            print(f"✅ Found answer in key '{key}': {answer[:100]}...")
                            break
                    
                    # Check if result contains a message object with content
                    if not answer and "result" in result:
                        result_value = result["result"]
                        # If result is a message object, try to extract content
                        if hasattr(result_value, "content"):
                            answer = result_value.content
                        elif isinstance(result_value, dict) and "content" in result_value:
                            answer = result_value["content"]
                        elif hasattr(result_value, "text"):
                            answer = result_value.text
                
                # If still empty, provide a fallback message
                if not answer:
                    print(f"⚠️  Still no answer found. Result structure: {result}")
                    answer = "I received your question but encountered an issue processing it. Please try rephrasing your question."
            
            tool_calls_executed = []
            
            # Extract source information
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
            
            if debug:
                print(f"💬 LLM Response: {answer}")
                print(f"📚 Sources used: {len(sources)}")
                if tool_calls_executed:
                    print(f"🔧 Tools executed: {tool_calls_executed}")
            
            response = {
                "answer": answer,
                "sources": sources,
                "error": False
            }
            
            if tool_calls_executed:
                response["tools_called"] = tool_calls_executed
            
            return response
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"❌ Error in ask method: {str(e)}")
            if debug:
                print(f"Traceback:\n{error_trace}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "error": True
            }
    
    def _format_question_with_history(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Format question with conversation history context
        
        Args:
            question: Current question
            conversation_history: List of previous Q&A pairs
            
        Returns:
            Formatted question with context
        """
        if not conversation_history or len(conversation_history) == 0:
            return question
        
        # Build context from recent conversation history (last 5 exchanges)
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        
        context_parts = ["Previous conversation context:"]
        for i, exchange in enumerate(recent_history, 1):
            prev_q = exchange.get("question", "")
            prev_a = exchange.get("answer", "")
            if prev_q and prev_a:
                context_parts.append(f"Q{i}: {prev_q}")
                context_parts.append(f"A{i}: {prev_a}")
        
        context_parts.append(f"\nCurrent question: {question}")
        
        return "\n".join(context_parts)
    
    def _create_address_verification_tool(self) -> StructuredTool:
        """
        Create a LangChain tool from AddressVerificationTool
        
        Returns:
            StructuredTool that can be used by LangChain LLM
        """
        class AddressVerificationInput(BaseModel):
            address: str = Field(description="The full address string to verify")
            zipcode: str = Field(description="The 5-digit zipcode to verify")
            county: Optional[str] = Field(default=None, description="Optional county name (e.g., 'Galveston County')")
        
        def verify_address(address: str, zipcode: str, county: Optional[str] = None) -> str:
            """
            Verify if an address is valid for TWIA coverage.
            Only validates addresses in 14 Texas coastal counties or portions of Harris County east of Highway 146.
            
            Args:
                address: The full address string
                zipcode: The 5-digit zipcode
                county: Optional county name
                
            Returns:
                JSON string with verification results
            """
            import json
            result = self.address_verifier.verify(address=address, zipcode=zipcode, county=county)
            return json.dumps(result, indent=2)
        
        return StructuredTool.from_function(
            func=verify_address,
            name="verify_address",
            description="""Verify if an address is eligible for TWIA (Texas Windstorm Insurance Association) coverage.
            This tool checks if an address is located in one of the 14 designated Texas coastal counties 
            or in portions of Harris County east of Highway 146. 
            Use this tool when users ask about address eligibility, coverage areas, or need to verify 
            if their property location qualifies for TWIA insurance.
            
            Input requires:
            - address: Full address string
            - zipcode: 5-digit zipcode
            - county: Optional county name (e.g., 'Galveston County', 'Harris County')
            
            Returns verification status and location details.""",
            args_schema=AddressVerificationInput
        )
    
    def diagnose_system(self):
        """
        Diagnose potential issues with the RAG system
        """
        print("\n🔧 DIAGNOSTIC REPORT")
        print("=" * 50)
        
        # Check if vector store exists and has data
        if not self.vector_store:
            print("❌ Vector store not initialized")
            return
        
        try:
            # Get collection info
            collection = self.vector_store._collection
            count = collection.count()
            print(f"📊 Documents in vector store: {count}")
            
            if count == 0:
                print("❌ No documents in vector store!")
                print("💡 Try running 'rebuild' command")
                return
            
            # Test embedding
            test_query = "test query"
            test_results = self.vector_store.similarity_search(test_query, k=2)
            print(f"🔍 Test retrieval returned: {len(test_results)} documents")
            
            if test_results:
                print(f"📄 Sample retrieved content (first 100 chars):")
                print(f"   '{test_results[0].page_content[:100]}...'")
            
            # Check embedding model
            print(f"🧠 Embedding model: {self.embedding_function.model_name}")
            
            # Test LLM connection
            try:
                llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano")
                test_response = llm.invoke("Say 'Hello'")
                print(f"🤖 LLM connection: ✅ Working")
            except Exception as e:
                print(f"🤖 LLM connection: ❌ Error - {str(e)}")
            
            # Check QA chain
            if self.qa_chain:
                print(f"🔗 QA Chain: ✅ Initialized")
            else:
                print(f"🔗 QA Chain: ❌ Not initialized")
            
        except Exception as e:
            print(f"❌ Diagnostic error: {str(e)}")
        
        print("=" * 50 + "\n")
    
    def test_retrieval(self, question: str, k: int = 5):
        """
        Test document retrieval without LLM
        """
        if not self.vector_store:
            print("Vector store not initialized")
            return
        
        print(f"\n🔍 TESTING RETRIEVAL")
        print(f"Question: '{question}'")
        print("-" * 30)
        
        try:
            results = self.vector_store.similarity_search_with_score(question, k=k)
            
            if not results:
                print("❌ No documents retrieved!")
                return
            
            for i, (doc, score) in enumerate(results):
                print(f"\nDoc {i+1} (Score: {score:.4f}):")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
                
        except Exception as e:
            print(f"❌ Retrieval error: {str(e)}")
    
    def rebuild_vector_store(self):
        """
        Force rebuild the vector store
        """
        print("🔄 Rebuilding vector store...")
        
        # Remove existing vector store
        import shutil
        if os.path.exists(self.vector_db_path):
            shutil.rmtree(self.vector_db_path)
        
        # Reload and rebuild
        self.setup_rag_system()
    
    def check_document_quality(self):
        """
        Check the quality of loaded documents
        """
        if not self.vector_store:
            print("Vector store not initialized")
            return
        
        print("\n📋 DOCUMENT QUALITY CHECK")
        print("=" * 40)
        
        try:
            # Get a sample of documents
            sample_docs = self.vector_store.similarity_search("sample", k=10)
            
            if not sample_docs:
                print("❌ No documents found")
                return
            
            total_chars = sum(len(doc.page_content) for doc in sample_docs)
            avg_length = total_chars / len(sample_docs)
            
            print(f"📊 Sample size: {len(sample_docs)} chunks")
            print(f"📏 Average chunk length: {avg_length:.0f} characters")
            
            # Check for very short documents
            short_docs = [doc for doc in sample_docs if len(doc.page_content) < 100]
            if short_docs:
                print(f"⚠️  Warning: {len(short_docs)} very short chunks found")
            
            # Check for very long documents  
            long_docs = [doc for doc in sample_docs if len(doc.page_content) > 2000]
            if long_docs:
                print(f"⚠️  Warning: {len(long_docs)} very long chunks found")
            
            # Show sample content
            print(f"\n📄 Sample document content:")
            print(f"'{sample_docs[0].page_content[:200]}...'")
            
        except Exception as e:
            print(f"❌ Quality check error: {str(e)}")
        
        print("=" * 40 + "\n")
    
    def add_pdfs(self, new_pdf_paths: List[str]):
        """
        Add new PDF files and rebuild the vector store
        
        Args:
            new_pdf_paths: List of paths to new PDF files
        """
        print("Adding new PDF files...")
        
        # Copy new PDFs to the PDF directory
        for pdf_path in new_pdf_paths:
            if os.path.exists(pdf_path):
                import shutil
                filename = os.path.basename(pdf_path)
                destination = os.path.join(self.pdf_directory, filename)
                shutil.copy2(pdf_path, destination)
                print(f"Added: {filename}")
        
        # Rebuild the RAG system
        print("Rebuilding RAG system with new documents...")
        self.rebuild_vector_store()
    
    def evaluate_system_simple(self, test_questions: List[str], expected_keywords: List[List[str]] = None):
        """
        Simple evaluation without RAGAS (to avoid dependency conflicts)
        
        Args:
            test_questions: List of test questions
            expected_keywords: List of keywords that should appear in answers (optional)
        """
        if not self.qa_chain:
            print("RAG system not initialized")
            return {}
        
        print("🔍 Evaluating RAG system (Simple)...")
        print("=" * 40)
        
        results = []
        for i, question in enumerate(test_questions):
            print(f"\nQuestion {i+1}: {question}")
            response = self.ask(question)
            
            if response["error"]:
                print(f"❌ Error: {response['answer']}")
                results.append({
                    "question": question,
                    "answer": response["answer"],
                    "sources_count": 0,
                    "has_keywords": False
                })
            else:
                print(f"✅ Answer: {response['answer'][:100]}...")
                print(f"📚 Sources: {len(response['sources'])}")
                
                # Check for expected keywords if provided
                has_keywords = True
                if expected_keywords and i < len(expected_keywords):
                    keywords = expected_keywords[i]
                    answer_lower = response["answer"].lower()
                    has_keywords = any(keyword.lower() in answer_lower for keyword in keywords)
                    print(f"🔍 Keywords found: {has_keywords}")
                
                results.append({
                    "question": question,
                    "answer": response["answer"],
                    "sources_count": len(response["sources"]),
                    "has_keywords": has_keywords
                })
        
        # Summary
        print("\n📊 EVALUATION SUMMARY")
        print("=" * 30)
        total_questions = len(results)
        answered_questions = len([r for r in results if not r["answer"].startswith("Error")])
        sourced_answers = len([r for r in results if r["sources_count"] > 0])
        keyword_matches = len([r for r in results if r["has_keywords"]])
        
        print(f"Total questions: {total_questions}")
        print(f"Successfully answered: {answered_questions} ({answered_questions/total_questions*100:.1f}%)")
        print(f"Answers with sources: {sourced_answers} ({sourced_answers/total_questions*100:.1f}%)")
        if expected_keywords:
            print(f"Keyword matches: {keyword_matches} ({keyword_matches/total_questions*100:.1f}%)")
        
        return results
    
    def evaluate_system(self, test_questions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """
        Evaluate the RAG system using RAGAS metrics
        
        Args:
            test_questions: List of test questions
            ground_truths: List of expected answers
            
        Returns:
            Dictionary of evaluation scores
        """
        if not self.qa_chain:
            print("RAG system not initialized")
            return {}
        
        print("Evaluating RAG system...")
        
        results = []
        contexts = []
        
        for question in test_questions:
            result = self.qa_chain.invoke({"query": question})
            results.append(result['result'])
            
            # Extract contexts
            sources = result["source_documents"]
            contents = [source.page_content for source in sources]
            contexts.append(contents)
        
        # Create dataset for evaluation
        eval_data = {
            "question": test_questions,
            "answer": results,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(eval_data)
        
        # Wrap embeddings for RAGAS compatibility
        # Use OpenAI embeddings for RAGAS as it's more compatible
        from langchain_openai import OpenAIEmbeddings
        # Configure OpenAI embeddings with timeout and retry settings
        ragas_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                request_timeout=120,  # Increase timeout to 120 seconds
                max_retries=3,  # Retry up to 3 times on failure
            )
        )
        
        # Evaluate using RAGAS metrics with error handling
        print("Note: Evaluation may take several minutes. Timeout set to 120 seconds per request.")
        try:
            score = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    context_entity_recall,
               #     answer_similarity,
                    answer_correctness
                    #BleuScore
         #           harmfulness
                ],
                embeddings=ragas_embeddings
            )
        except TimeoutError as e:
            print(f"❌ RAGAS evaluation timed out: {e}")
            print("💡 Try: Increasing timeout, reducing number of metrics, or checking network connection")
            raise
        except Exception as e:
            print(f"❌ Error during RAGAS evaluation: {e}")
            print("💡 Check your OpenAI API key and network connection")
            raise
        
        # Save results
        score_df = score.to_pandas()
        filenameEval = "EvaluationScores_" + date.today().strftime("%Y%m%d") + ".csv"
        score_df.to_csv(filenameEval, encoding="utf-8", index=False)
        
        # Return mean scores
        mean_scores = score_df[[
            'faithfulness', 'answer_relevancy', 'context_precision',
            'context_recall', 'context_entity_recall',
            'answer_correctness', # 'harmfulness' 'answer_similarity',
        ]].mean(axis=0).to_dict()
        
        print("Evaluation completed. Results saved to EvaluationScores.csv")
        return mean_scores
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n" + "="*50)
        print("PDF RAG Agent - Interactive Chat")
        print("="*50)
        print("Ask questions about your PDF documents!")
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Type 'help' for available commands")
        print("-"*50 + "\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("Agent: Goodbye! Thanks for using PDF RAG Agent.")
                    break
                
                if question.lower() == 'help':
                    print("""
Available commands:
- Ask any question about your PDF documents
- 'diagnose': Run system diagnostics
- 'test <question>': Test retrieval without LLM
- 'rebuild': Rebuild vector store
- 'quality': Check document quality
- 'debug <question>': Ask question with debug info
- 'quit', 'exit', 'bye': End the session
- 'help': Show this help message
                    """)
                    continue
                
                if question.lower() == 'diagnose':
                    self.diagnose_system()
                    continue
                
                if question.lower().startswith('test '):
                    test_q = question[5:].strip()
                    if test_q:
                        self.test_retrieval(test_q)
                    else:
                        print("Usage: test <your question>")
                    continue
                
                if question.lower() == 'rebuild':
                    self.rebuild_vector_store()
                    continue
                
                if question.lower() == 'quality':
                    self.check_document_quality()
                    continue
                
                if question.lower().startswith('debug '):
                    debug_q = question[6:].strip()
                    if debug_q:
                        response = self.ask(debug_q, debug=True)
                        if response["error"]:
                            print(f"Agent: ❌ {response['answer']}")
                        else:
                            print(f"Agent: {response['answer']}")
                    else:
                        print("Usage: debug <your question>")
                    continue
                
                if not question:
                    continue
                
                print("Agent: Thinking...")
                response = self.ask(question)
                
                if response["error"]:
                    print(f"Agent: ❌ {response['answer']}")
                else:
                    print(f"Agent: {response['answer']}")
                    
                    if response["sources"]:
                        print(f"\n📚 Sources ({len(response['sources'])} documents referenced)")
                        for i, source in enumerate(response["sources"][:2]):  # Show first 2 sources
                            print(f"  Source {i+1}: {source['content'][:100]}...")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\nAgent: Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Agent: ❌ An error occurred: {str(e)}")


def main():
    """Main function to run the PDF RAG Agent"""
    parser = argparse.ArgumentParser(
        description="PDF RAG Agent - Query PDF documents using RAG (Retrieval Augmented Generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directories (ragData and vectorRAGdb)
  python newcbseTest.py
  
  # Specify custom PDF directory
  python newcbseTest.py --pdf-dir ./custom_pdfs
  
  # Specify custom PDF directory and vector DB path
  python newcbseTest.py --pdf-dir ./custom_pdfs --vector-db ./custom_vector_db
        """
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Directory containing PDF files (default: llm-compliance/ragData)"
    )
    parser.add_argument(
        "--vector-db",
        type=str,
        default=None,
        help="Path to vector database storage (default: llm-compliance/vectorRAGdb)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting PDF RAG Agent...")
    if args.pdf_dir:
        print(f"📁 Using PDF directory: {args.pdf_dir}")
    if args.vector_db:
        print(f"💾 Using vector DB path: {args.vector_db}")
    
    # Initialize the agent with provided or default paths
    #agent = PDFRAGAgent(pdf_directory=args.pdf_dir, vector_db_path=args.vector_db)
    agent = PDFRAGAgent()
    
    # Start interactive chat
    #agent.diagnose_system()
    #llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    #print(llm.invoke("hi"))
    #retriever = agent.vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    #docs = retriever.get_relevant_documents("tell me about 'Sundarbans: Home of the Mangroves'")
    #for doc in docs:
    #    print(doc.page_content)
    agent.interactive_chat()
        # Example test questions and ground truths
    test_questions = [
        "what is twia?",
        "what are the covered perils?",
        "what are the covered property?",
        "what are the coverage options?",
        "what are the optional endorsements?",
    ]
    ground_truths = [
        "twia is a state-sponsored insurance program for windstorm and hail damage in coastal counties of Texas.",
        "covered perils include windstorm and hail damage",
        "covered property includes dwelling and other structures including your primary residence and structures on your property such as sheds or detached garages. Personal property including personal belongings is covered with different settlement options available. Builder's risk endorsements can be added for buildings and structures under construction. Endorsements are also available to provide coverage for manufactured homes.",
        "coverage options include replacement cost (RCV) where if your insurance amount is 80% or more of the dwelling's full replacement cost, you are paid the full cost of repairs or replacement without a deduction for depreciation. Actual Cash Value (ACV) where if your insurance amount is less than 80% of the replacement cost, the payment will not exceed the replacement cost, minus depreciation. Actual Cash Value Roof endorsement can specifically limit roof coverage to ACV, which factors in depreciation.",
        "optional endorsements include additional living expense (ALE) endorsement TWIA 310/320 covers up to 20% of your dwelling limit for extra living expenses if you are displaced from your home due to a covered loss. Increased Cost of Construction (ICC) endorsement covers the additional cost of bringing a damaged property up to current building codes during repairs. Indirect (Consequential) Loss endorsement TWIA 310/320/330 can cover personal property damage resulting from temperature changes, but only if it is a direct consequence of a covered loss."
    ]

    # Call the evaluation function
    results = agent.evaluate_system(test_questions, ground_truths)
    print("Evaluation Results:", results)
    
    # Pytest assertions
    assert results is not None, "Results should not be None"
    assert isinstance(results, dict), "Results should be a dictionary"
    assert len(results) > 0, "Results should not be empty"
    
    # Define thresholds for each metric
    thresholds = {
        'faithfulness': 0.7,           # Answers should be faithful to context
        'answer_relevancy': 0.6,       # Answers should be relevant to questions
        'context_precision': 0.5,      # Retrieved contexts should be precise
        'context_recall': 0.5,         # Should recall relevant contexts
        'context_entity_recall': 0.4,  # Should recall important entities
        'answer_correctness': 0.6      # Answers should be correct
    }
    
    # Check for expected metric keys and assert against thresholds
    for metric, threshold in thresholds.items():
        assert metric in results, f"Missing metric: {metric}"
        assert 0 <= results[metric] <= 1, f"{metric} score should be between 0 and 1, got {results[metric]}"
        assert results[metric] >= threshold, \
            f"{metric} score {results[metric]:.3f} is below threshold {threshold} ❌"
        print(f"✅ {metric}: {results[metric]:.3f} (threshold: {threshold})")
    
    print("✅ All assertions passed!")
    agent.interactive_chat()

def create_sample_documents() -> List[Document]:
    """Create sample documents for RAG based on TWIA information"""
    documents = [
        Document(
            page_content="""TWIA Overview and Purpose
The Texas Windstorm Insurance Association (TWIA) offers coverage for windstorm and hail damage for properties in designated coastal counties that have been denied coverage by private insurers. As a "last resort" insurer, TWIA policies only cover specific perils and must be combined with a standard homeowners policy for other protections. TWIA serves as a safety net for property owners who cannot obtain windstorm coverage in the private market.""",
            metadata={"source": "twia_overview", "type": "insurance", "category": "general"}
        ),
        Document(
            page_content="""TWIA Covered Perils - Wind and Hail Damage
The core of a TWIA policy is protection against wind and hail damage. This primary coverage pays for damage to the structure of your home, other buildings, and personal belongings caused by wind or hail. Wind-driven rain coverage can be extended through endorsement TWIA 320, which covers damage caused by rain that enters the property as a direct result of wind or hail damage.""",
            metadata={"source": "twia_covered_perils", "type": "insurance", "category": "coverage"}
        ),
        Document(
            page_content="""TWIA Covered Property - Structures and Personal Property
TWIA covers dwelling and other structures including your primary residence and structures on your property such as sheds or detached garages. Personal property including personal belongings is covered with different settlement options available. Builder's risk endorsements can be added for buildings and structures under construction. Endorsements are also available to provide coverage for manufactured homes.""",
            metadata={"source": "twia_covered_property", "type": "insurance", "category": "coverage"}
        ),
        Document(
            page_content="""TWIA Standard Coverage Options - RCV and ACV
TWIA offers different levels of coverage: Replacement Cost (RCV) where if your insurance amount is 80% or more of the dwelling's full replacement cost, you are paid the full cost of repairs or replacement without a deduction for depreciation. Actual Cash Value (ACV) where if your insurance amount is less than 80% of the replacement cost, the payment will not exceed the replacement cost, minus depreciation. Actual Cash Value Roof endorsement can specifically limit roof coverage to ACV, which factors in depreciation.""",
            metadata={"source": "twia_coverage_options", "type": "insurance", "category": "coverage"}
        ),
        Document(
            page_content="""TWIA Optional Coverage Endorsements - ALE, ICC, and Indirect Loss
Policyholders can increase their protection by adding specific endorsements. Additional Living Expense (ALE) endorsement TWIA 310/320 covers up to 20% of your dwelling limit for extra living expenses if you are displaced from your home due to a covered loss. Increased Cost of Construction (ICC) endorsement covers the additional cost of bringing a damaged property up to current building codes during repairs. Indirect (Consequential) Loss endorsement TWIA 310/320/330 can cover personal property damage resulting from temperature changes, but only if it is a direct consequence of a covered loss.""",
            metadata={"source": "twia_endorsements", "type": "insurance", "category": "options"}
        ),
        Document(
            page_content="""TWIA Exclusions - What is Not Covered
TWIA specifically excludes flood damage. In certain flood zones, having flood insurance is a prerequisite for a TWIA policy. Other perils not covered include damages from causes like fire, theft, and water damage from sources other than wind-driven rain. Since November 2013, metal screen enclosures and their contents are not covered unless specifically scheduled on the declarations page. Structures built in Coastal Barrier Resource Act (COBRA) zones after the grandfather date are generally ineligible for coverage.""",
            metadata={"source": "twia_exclusions", "type": "insurance", "category": "limitations"}
        ),
        Document(
            page_content="""TWIA Coverage Requirements - Need for Standard Homeowners Policy
It is crucial to know that TWIA must be combined with a standard homeowners policy that covers other perils not included in TWIA coverage. TWIA policies only cover specific perils and property owners need both TWIA windstorm coverage and a standard homeowners policy for comprehensive protection.""",
            metadata={"source": "twia_requirements", "type": "insurance", "category": "requirements"}
        ),
        Document(
            page_content="""TWIA Deductibles - Windstorm and Hail Coverage
TWIA policies have specific deductible structures for windstorm and hail damage. The deductible is typically a percentage of the dwelling coverage amount, commonly ranging from 1% to 5% of the coverage limit. For example, if your dwelling is insured for $300,000 with a 2% deductible, you would pay $6,000 out of pocket before TWIA coverage applies. Deductibles may vary based on the property's location, construction type, and specific policy terms. Higher deductibles can result in lower premiums, but policyholders should ensure they can afford the deductible amount in the event of a claim.""",
            metadata={"source": "twia_deductibles", "type": "insurance", "category": "coverage"}
        ),
        Document(
            page_content="""TWIA Eligibility Requirements - Property and Location
To be eligible for TWIA coverage, properties must be located in one of the 14 designated Texas coastal counties or portions of Harris County. The property must have been denied windstorm coverage by at least one private insurance company. Properties must meet specific building code requirements and may require a Windstorm Certificate of Compliance (WPI-8) issued by the Texas Department of Insurance. New construction or substantially improved properties must comply with current building codes. Properties in Coastal Barrier Resource Act (COBRA) zones may have additional restrictions or be ineligible depending on construction dates.""",
            metadata={"source": "twia_eligibility", "type": "insurance", "category": "requirements"}
        ),
        Document(
            page_content="""TWIA Claims Process - Filing and Settlement
When filing a TWIA claim, policyholders should report damage as soon as possible after a windstorm or hail event. The claims process typically involves: documenting all damage with photos and videos, preventing further damage with temporary repairs (keep receipts), filing a claim with TWIA, scheduling an inspection with a TWIA adjuster, receiving a claim estimate, and working with contractors for repairs. TWIA may issue advance payments for emergency repairs or additional living expenses. Policyholders should maintain detailed records of all damage, repairs, and expenses. The claims settlement process follows the policy's coverage terms, including applicable deductibles and coverage limits.""",
            metadata={"source": "twia_claims", "type": "insurance", "category": "claims"}
        ),
        Document(
            page_content="""TWIA Policy Limits and Coverage Amounts
TWIA coverage limits are determined based on the property's replacement cost value. Dwelling coverage limits should be set at or near the full replacement cost of the structure to ensure adequate protection. Personal property coverage is typically a percentage of the dwelling coverage, often 40% to 50%. Other structures coverage is usually limited to 10% of the dwelling coverage amount. Policyholders should regularly review and update their coverage limits to account for inflation, home improvements, or changes in property value. Underinsuring a property can result in coinsurance penalties and reduced claim payments.""",
            metadata={"source": "twia_limits", "type": "insurance", "category": "coverage"}
        ),
        Document(
            page_content="""TWIA Windstorm Certification Requirements
Properties in TWIA-eligible areas may require a Windstorm Certificate of Compliance (WPI-8) to obtain or maintain coverage. This certificate verifies that the property meets current windstorm building code requirements. New construction, substantial improvements, or properties that have undergone significant renovations typically require inspection and certification. The certificate must be obtained from the Texas Department of Insurance or an authorized inspector. Properties without a valid certificate may be denied coverage or have coverage limitations. The certificate demonstrates compliance with building codes designed to withstand hurricane-force winds common in coastal Texas areas.""",
            metadata={"source": "twia_certification", "type": "insurance", "category": "requirements"}
        ),
        Document(
            page_content="""TWIA Premiums and Payment Options
TWIA premiums are calculated based on several factors including property location, construction type, coverage amounts, deductibles selected, and property characteristics. Premiums may be paid annually, semi-annually, or through installment plans. Policyholders can typically pay by check, credit card, or electronic funds transfer. Late payment may result in policy cancellation or non-renewal. TWIA premiums are generally higher than standard homeowners insurance due to the high-risk nature of coastal windstorm exposure. Policyholders should compare quotes and understand that TWIA serves as a last resort when private market coverage is unavailable.""",
            metadata={"source": "twia_premiums", "type": "insurance", "category": "pricing"}
        ),
        Document(
            page_content="""TWIA Policy Renewal and Cancellation
TWIA policies are typically written for one-year terms and must be renewed annually. Policyholders will receive a renewal notice before the policy expiration date. Renewal may require updated property information, proof of continued eligibility, and payment of the renewal premium. TWIA may non-renew policies for various reasons including non-payment, property changes that affect eligibility, or failure to maintain required certifications. Policyholders can cancel their TWIA policy at any time, though cancellation may require written notice. If a policy is cancelled mid-term, premium refunds are typically calculated on a pro-rata basis minus any applicable fees.""",
            metadata={"source": "twia_renewal", "type": "insurance", "category": "policy"}
        ),
        Document(
            page_content="""TWIA Loss Settlement Options - RCV vs ACV
TWIA offers two primary loss settlement methods: Replacement Cost Value (RCV) and Actual Cash Value (ACV). Under RCV settlement, if the dwelling is insured to at least 80% of its replacement cost, TWIA will pay the full cost to repair or replace damaged property without deducting for depreciation. Under ACV settlement, payments are based on replacement cost minus depreciation, which reflects the property's age and condition. The settlement method affects how claims are paid and policyholders should understand which method applies to their coverage. Roof coverage may have specific ACV limitations even when the dwelling has RCV coverage.""",
            metadata={"source": "twia_settlement", "type": "insurance", "category": "claims"}
        ),
        Document(
            page_content="""TWIA Personal Property Coverage Details
TWIA personal property coverage protects belongings inside the insured dwelling, including furniture, electronics, clothing, appliances, and other personal items. Coverage typically applies to 40% to 50% of the dwelling coverage amount. Personal property is subject to the same windstorm and hail perils as the dwelling. Some items may have special limits or require scheduling for full coverage, such as jewelry, fine arts, or collectibles. Personal property coverage may have different settlement options (RCV or ACV) depending on the policy terms. Policyholders should maintain an inventory of personal property to facilitate claims processing.""",
            metadata={"source": "twia_personal_property", "type": "insurance", "category": "coverage"}
        ),
        Document(
            page_content="""TWIA Other Structures Coverage
TWIA covers other structures on the insured property that are not attached to the main dwelling, such as detached garages, sheds, fences, gazebos, and storage buildings. Coverage for other structures is typically limited to 10% of the dwelling coverage amount. These structures must be on the same property as the insured dwelling and used for residential purposes. Coverage applies to the same windstorm and hail perils as the main dwelling. Policyholders should ensure their other structures coverage limit is adequate to replace or repair detached structures on their property.""",
            metadata={"source": "twia_other_structures", "type": "insurance", "category": "coverage"}
        ),
        Document(
            page_content="""TWIA Dwelling Coverage - Structure Protection
TWIA dwelling coverage protects the main structure of your home, including the foundation, walls, roof, built-in appliances, and permanently installed fixtures. Coverage extends to attached structures like attached garages, porches, and decks. The dwelling coverage amount should reflect the full replacement cost of the structure, not its market value. Coverage applies to damage caused by wind and hail, the primary perils covered by TWIA. Policyholders should regularly review their dwelling coverage to ensure it keeps pace with construction costs and home improvements that increase replacement value.""",
            metadata={"source": "twia_dwelling", "type": "insurance", "category": "coverage"}
        ),
        Document(
            page_content="""TWIA Building Code Requirements and Compliance
Properties covered by TWIA must meet specific building code requirements designed to withstand hurricane-force winds. These requirements include proper roof attachment, window and door protection, structural connections, and foundation anchoring. New construction or substantially improved properties must comply with current International Building Code (IBC) or International Residential Code (IRC) standards for wind resistance. Properties may require a Windstorm Certificate of Compliance (WPI-8) to demonstrate code compliance. Failure to meet building code requirements can result in coverage denial, policy cancellation, or reduced claim payments if code violations contributed to damage.""",
            metadata={"source": "twia_building_codes", "type": "insurance", "category": "requirements"}
        ),
        Document(
            page_content="""TWIA Geographic Coverage Areas
TWIA coverage is available in 14 designated Texas coastal counties: Aransas, Brazoria, Calhoun, Cameron, Chambers, Galveston, Jefferson, Kenedy, Kleberg, Matagorda, Nueces, Refugio, San Patricio, and Willacy. Coverage is also available in portions of Harris County that meet specific eligibility criteria. Properties must be located within these designated areas to qualify for TWIA coverage. The association was created to provide windstorm insurance in areas where private insurers are unwilling or unable to provide coverage due to high hurricane risk. Property owners outside these areas are not eligible for TWIA coverage.""",
            metadata={"source": "twia_geographic", "type": "insurance", "category": "eligibility"}
        ),
        Document(
            page_content="""TWIA Manufactured Home Coverage
TWIA provides coverage for manufactured homes (mobile homes) located in eligible coastal areas. Manufactured home coverage requires the home to be properly anchored and meet specific installation and tie-down requirements. Coverage may be available for both the manufactured home structure and personal property inside. Special endorsements may be required for manufactured home coverage. The home must be located on a permanent foundation or properly secured to meet wind resistance standards. Manufactured homes must meet the same eligibility requirements as site-built homes, including location in a designated coastal county and denial of coverage by private insurers.""",
            metadata={"source": "twia_manufactured_homes", "type": "insurance", "category": "coverage"}
        ),
    ]
    return documents

class AddressVerificationTool:
    """Tool to verify address validity for 14 Texas coastal counties and portions of Harris County"""
    
    # 14 coastal counties in Texas
    TEXAS_COASTAL_COUNTIES = [
        {"name": "Aransas County", "state": "TX", "city": "Rockport"},
        {"name": "Brazoria County", "state": "TX", "city": "Angleton"},
        {"name": "Calhoun County", "state": "TX", "city": "Port Lavaca"},
        {"name": "Cameron County", "state": "TX", "city": "Brownsville"},
        {"name": "Chambers County", "state": "TX", "city": "Anahuac"},
        {"name": "Galveston County", "state": "TX", "city": "Galveston"},
        {"name": "Jefferson County", "state": "TX", "city": "Beaumont"},
        {"name": "Kenedy County", "state": "TX", "city": "Sarita"},
        {"name": "Kleberg County", "state": "TX", "city": "Kingsville"},
        {"name": "Matagorda County", "state": "TX", "city": "Bay City"},
        {"name": "Nueces County", "state": "TX", "city": "Corpus Christi"},
        {"name": "Refugio County", "state": "TX", "city": "Refugio"},
        {"name": "San Patricio County", "state": "TX", "city": "Sinton"},
        {"name": "Willacy County", "state": "TX", "city": "Raymondville"},
    ]
    
    # Portion of Harris County, Texas - Areas east of Highway 146
    # Sample zipcodes for areas east of Highway 146 (Baytown, La Porte, Seabrook, etc.)
    HARRIS_COUNTY_EAST = {
        "77520": {"zipcode": "77520", "city": "Baytown", "county": "Harris County", "state": "TX", "highway_146_east": True},
        "77521": {"zipcode": "77521", "city": "Baytown", "county": "Harris County", "state": "TX", "highway_146_east": True},
        "77546": {"zipcode": "77546", "city": "La Porte", "county": "Harris County", "state": "TX", "highway_146_east": True},
        "77547": {"zipcode": "77547", "city": "La Porte", "county": "Harris County", "state": "TX", "highway_146_east": True},
        "77587": {"zipcode": "77587", "city": "Seabrook", "county": "Harris County", "state": "TX", "highway_146_east": True},
        "77598": {"zipcode": "77598", "city": "Webster", "county": "Harris County", "state": "TX", "highway_146_east": True},
    }
    
    # Valid addresses with zipcodes for 14 Texas coastal counties
    VALID_ADDRESSES = {
        # Aransas County, TX
        "78382": {"zipcode": "78382", "city": "Rockport", "county": "Aransas County", "state": "TX"},
        "78383": {"zipcode": "78383", "city": "Fulton", "county": "Aransas County", "state": "TX"},
        
        # Brazoria County, TX
        "77515": {"zipcode": "77515", "city": "Angleton", "county": "Brazoria County", "state": "TX"},
        "77511": {"zipcode": "77511", "city": "Alvin", "county": "Brazoria County", "state": "TX"},
        
        # Calhoun County, TX
        "77979": {"zipcode": "77979", "city": "Port Lavaca", "county": "Calhoun County", "state": "TX"},
        "77957": {"zipcode": "77957", "city": "Seadrift", "county": "Calhoun County", "state": "TX"},
        
        # Cameron County, TX
        "78520": {"zipcode": "78520", "city": "Brownsville", "county": "Cameron County", "state": "TX"},
        "78521": {"zipcode": "78521", "city": "Brownsville", "county": "Cameron County", "state": "TX"},
        
        # Chambers County, TX
        "77514": {"zipcode": "77514", "city": "Anahuac", "county": "Chambers County", "state": "TX"},
        "77531": {"zipcode": "77531", "city": "Hankamer", "county": "Chambers County", "state": "TX"},
        
        # Galveston County, TX
        "77550": {"zipcode": "77550", "city": "Galveston", "county": "Galveston County", "state": "TX"},
        "77551": {"zipcode": "77551", "city": "Galveston", "county": "Galveston County", "state": "TX"},
        
        # Jefferson County, TX
        "77701": {"zipcode": "77701", "city": "Beaumont", "county": "Jefferson County", "state": "TX"},
        "77702": {"zipcode": "77702", "city": "Beaumont", "county": "Jefferson County", "state": "TX"},
        
        # Kenedy County, TX
        "78385": {"zipcode": "78385", "city": "Sarita", "county": "Kenedy County", "state": "TX"},
        
        # Kleberg County, TX
        "78363": {"zipcode": "78363", "city": "Kingsville", "county": "Kleberg County", "state": "TX"},
        "78351": {"zipcode": "78351", "city": "Riviera", "county": "Kleberg County", "state": "TX"},
        
        # Matagorda County, TX
        "77414": {"zipcode": "77414", "city": "Bay City", "county": "Matagorda County", "state": "TX"},
        "77978": {"zipcode": "77978", "city": "Palacios", "county": "Matagorda County", "state": "TX"},
        
        # Nueces County, TX
        "78401": {"zipcode": "78401", "city": "Corpus Christi", "county": "Nueces County", "state": "TX"},
        "78402": {"zipcode": "78402", "city": "Corpus Christi", "county": "Nueces County", "state": "TX"},
        
        # Refugio County, TX
        "78377": {"zipcode": "78377", "city": "Refugio", "county": "Refugio County", "state": "TX"},
        "78384": {"zipcode": "78384", "city": "Woodsboro", "county": "Refugio County", "state": "TX"},
        
        # San Patricio County, TX
        "78387": {"zipcode": "78387", "city": "Sinton", "county": "San Patricio County", "state": "TX"},
        "78361": {"zipcode": "78361", "city": "Ingleside", "county": "San Patricio County", "state": "TX"},
        
        # Willacy County, TX
        "78580": {"zipcode": "78580", "city": "Raymondville", "county": "Willacy County", "state": "TX"},
        "78586": {"zipcode": "78586", "city": "Sebastian", "county": "Willacy County", "state": "TX"},
    }
    
    # Merge Harris County east addresses into main VALID_ADDRESSES
    VALID_ADDRESSES.update(HARRIS_COUNTY_EAST)
    
    def __init__(self):
        self.name = "verify_address"
        self.description = """Verify if an address is valid based on zipcode and county name.
Input should be a dictionary with 'address', 'zipcode', and 'county' keys.
Returns a verification result with validity status and address details."""
    
    def verify(self, address: str, zipcode: str, county: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify address based on zipcode and county
        Only validates addresses in 14 Texas coastal counties or portions of Harris County east of Highway 146
        
        Args:
            address: The full address string
            zipcode: The zipcode to verify
            county: Optional county name to verify
            
        Returns:
            Dictionary with verification results including coastal status
        """
        # Check if zipcode exists in our database
        if zipcode not in self.VALID_ADDRESSES:
            # Check if county is in our Texas coastal counties list
            if county:
                county_normalized = county.lower().replace(" county", "").strip()
                coastal_match = False
                
                for coastal_county in self.TEXAS_COASTAL_COUNTIES:
                    if coastal_county["name"].lower() == county.lower() or coastal_county["name"].lower().replace(" county", "") == county_normalized:
                        coastal_match = True
                        break
                
                # Check if it's Harris County (special case for east of Highway 146)
                if county.lower() == "harris county":
                    return {
                        "valid": False,
                        "reason": f"Harris County zipcode {zipcode} is not in the area east of Highway 146",
                        "address": address,
                        "zipcode": zipcode,
                        "county": county,
                        "note": "Only areas east of Highway 146 in Harris County are valid"
                    }
                
                if coastal_match:
                    return {
                        "valid": True,
                        "message": f"Address verified in Texas coastal county: {county}",
                        "address": address,
                        "zipcode": zipcode,
                        "county": county,
                        "is_coastal": True,
                        "note": "County recognized but zipcode not in database"
                    }
            
            return {
                "valid": False,
                "reason": f"Zipcode {zipcode} not found in Texas coastal counties database",
                "address": address,
                "zipcode": zipcode,
                "county": county
            }
        
        address_data = self.VALID_ADDRESSES[zipcode]
        
        # Check county if provided
        if county and county.lower() not in address_data["county"].lower():
            return {
                "valid": False,
                "reason": f"County '{county}' does not match zipcode {zipcode}. Expected: {address_data['county']}",
                "address": address,
                "zipcode": zipcode,
                "expected_county": address_data["county"],
                "provided_county": county
            }
        
        # Determine if this is in Harris County east of Highway 146
        is_harris_east = zipcode in self.HARRIS_COUNTY_EAST
        
        # Address is valid in Texas coastal area
        return {
            "valid": True,
            "message": "Address verified in Texas coastal county or Harris County east of Highway 146",
            "address": address,
            "zipcode": zipcode,
            "city": address_data["city"],
            "county": address_data["county"],
            "state": address_data["state"],
            "is_coastal": True,
            "harris_highway_146_east": is_harris_east,
            "highway_146_east": address_data.get("highway_146_east", False)
        }


if __name__ == "__main__":
    main()