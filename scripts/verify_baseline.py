import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.models import Document
from src.pipeline.baseline import BaselineRAG

def verify_baseline():
    print("Initializing Baseline RAG Pipeline...")
    rag = BaselineRAG()
    
    # We will test with a small mock dataset first before trying the 500k files
    docs = [
        Document(
            doc_id="dsid_mock_1",
            source="slack",
            content="The production database will be down for maintenance from 2 AM to 4 AM EST on Sunday. Contact devops if you have questions.",
            metadata={"author": "alice"}
        ),
        Document(
            doc_id="dsid_mock_2",
            source="github",
            content="PR #5012: Added Qdrant as our vector database for the Baseline RAG system. It runs locally and supports up to 1M vectors easily.",
            metadata={"author": "bob"}
        )
    ]
    
    print("Ingesting mock documents into Qdrant...")
    rag.ingest_documents(docs)
    
    queries = [
        "When is the production database maintenance?",
        "What vector database are we using for Baseline RAG?"
    ]
    
    for q in queries:
        print(f"\n--- Query: {q} ---")
        try:
            answer = rag.ask(q, top_k=2)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Failed to generate answer. Is the LLM running? Error: {e}")

if __name__ == "__main__":
    verify_baseline()
