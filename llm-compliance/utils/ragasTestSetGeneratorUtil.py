import json
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)
# Or use the default synthesizers
from ragas.testset.synthesizers import default_query_distribution
# For document preparation
from langchain_core.documents import Document
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import dotenv
import os
dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Get path to ragData directory relative to this file
project_root = Path(__file__).parent.parent
rag_data_path = project_root / "ragData"

loader = DirectoryLoader(
    str(rag_data_path), use_multithreading=True, silent_errors=True, sample_size=1
)
documents = loader.load()

for document in documents:
    document.metadata["filename"] = document.metadata["source"]


# Use HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

generator = TestsetGenerator.from_langchain(
 #   generator_llm=azure_model, critic_llm=azure_model, embeddings=azure_embeddings
    llm=OpenAI(model="gpt-4o-mini"), embedding_model=embeddings
)



# Specify distribution when generating
testset = generator.generate_with_langchain_docs(
    documents=documents,
    testset_size=10,
    query_distribution={
        SingleHopSpecificQuerySynthesizer: 0.5,
        MultiHopAbstractQuerySynthesizer: 0.3,
        MultiHopSpecificQuerySynthesizer: 0.2
    }
)

print(testset)

# Convert to dictionary format
testset_dict = testset.to_pandas().to_dict(orient='records')

# Save to JSON file
with open('testset.json', 'w', encoding='utf-8') as f:
    json.dump(testset_dict, f, indent=2, ensure_ascii=False)
