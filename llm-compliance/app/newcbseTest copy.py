from datetime import date
import os
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
        if pdf_directory is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)  # Go up to llm-compliance/ directory
            pdf_directory = os.path.join(parent_dir, "ragData")
        
        # Default to vectorragdb directory if not specified
        if vector_db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)  # Go up to llm-compliance/ directory
            vector_db_path = os.path.join(parent_dir, "vectorRAGdb")
        
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
        
        # Create directories if they don't exist
        os.makedirs(pdf_directory, exist_ok=True)
        
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
            prompt_config_path = os.path.join(current_dir, "ragSysPrompts", "leniant_rag_system.yaml")
        
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
        You are a helpful AI assistant. Answer the question using ONLY the information provided in the context below. 
        
        IMPORTANT RULES:
        1. Base your answer STRICTLY on the provided context
        2. If the context doesn't contain information to answer the question, respond with: "I cannot find information about this question in the provided documents."
        3. Do not use your general knowledge - only use the context provided
        4. Quote relevant parts from the context when possible
        5. Be specific and cite which document or section the information comes from when available
        
        Context from documents:
        {context}
        
        Question: {question}
        
        Answer based strictly on the context above:
        """
        except Exception as e:
            print(f"⚠️  Error loading prompt config: {e}")
            print("   Using default prompt template...")
            # Fallback to default prompt
            PROMPT_TEMPLATE = """
        You are a helpful AI assistant. Answer the question using ONLY the information provided in the context below. 
        
        IMPORTANT RULES:
        1. Base your answer STRICTLY on the provided context
        2. If the context doesn't contain information to answer the question, respond with: "I cannot find information about this question in the provided documents."
        3. Do not use your general knowledge - only use the context provided
        4. Quote relevant parts from the context when possible
        5. Be specific and cite which document or section the information comes from when available
        
        Context from documents:
        {context}
        
        Question: {question}
        
        Answer based strictly on the context above:
        """
        
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    temperature=0, 
                    model="gpt-3.5-turbo",
                    max_tokens=500  # Limit response length to control costs (reduces output token usage by 20-30%)
                ),
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        'k': 2  # Reduced from 5 to 3: Retrieve top 3 most relevant chunks (reduces input token usage by 20-30%)
                        #'fetch_k': 20  # Fetch more candidates before filtering
                    }
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)}
            )
            print("QA chain setup completed")
        except Exception as e:
            print(f"Error setting up QA chain: {e}")
    
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
        print("📚 Loading PDFs and creating vector database...")
        documents = self.load_pdfs()
        
        if not documents:
            print("No documents loaded. RAG system not initialized.")
            return
        
        processed_docs = self.process_documents(documents)
        self.create_vector_store(processed_docs)
        self.setup_qa_chain()
        
        if self.qa_chain:
            print("✅ RAG system initialized successfully!")
        else:
            print("❌ RAG system initialization failed!")
    
    def ask(self, question: str, debug: bool = False) -> Dict[str, Any]:
        """
        Ask a question to the RAG system
        
        Args:
            question: The question to ask
            debug: If True, print debug information
            
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
            # Debug: Test retrieval directly
            if debug:
                print(f"\n🔍 DEBUG: Testing retrieval for question: '{question}'")
                retrieved_docs = self.vector_store.similarity_search(question, k=3)  # Match QA chain retrieval count
                print(f"📄 Retrieved {len(retrieved_docs)} documents:")
                for i, doc in enumerate(retrieved_docs):
                    print(f"  Doc {i+1}: {doc.page_content[:100]}...")
                    print(f"  Metadata: {doc.metadata}")
                print("-" * 50)
            
            result = self.qa_chain.invoke({"query": question})
            
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
                print(f"💬 LLM Response: {result['result']}")
                print(f"📚 Sources used: {len(sources)}")
            
            return {
                "answer": result["result"],
                "sources": sources,
                "error": False
            }
        
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "error": True
            }
    
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
            test_results = self.vector_store.similarity_search(test_query, k=3)
            print(f"🔍 Test retrieval returned: {len(test_results)} documents")
            
            if test_results:
                print(f"📄 Sample retrieved content (first 100 chars):")
                print(f"   '{test_results[0].page_content[:100]}...'")
            
            # Check embedding model
            print(f"🧠 Embedding model: {self.embedding_function.model_name}")
            
            # Test LLM connection
            try:
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
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
    agent.diagnose_system()
    #llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    #print(llm.invoke("hi"))
    #retriever = agent.vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    #docs = retriever.get_relevant_documents("tell me about 'Sundarbans: Home of the Mangroves'")
    #for doc in docs:
    #    print(doc.page_content)
    #agent.interactive_chat()
        # Example test questions and ground truths
    test_questions = [
        "What is the Sundarbans?",
        "where godavari river started?",
        "explain water life cycle"
    ]
    ground_truths = [
        "The Sundarbans is the largest mangrove forest in the world. It is located at delta formed by the confluence of the Ganges, Brahmaputra and Meghna Rivers in the Bay of Bengal.",
        "The Godavari river starts in the Western Ghats at Trimbakeshwar in Maharashtra.(Chapter 2: Journey of a River)",
        "heat causes water from different sources, like oceans and rivers, to become water vapour. Water vapour forms the clouds, which come down as rain, snow and hail. This water goes back into rivers, lakes and oceans.This constant circular movement of water in nature is called the water cycle."
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



if __name__ == "__main__":
    main()