import sys
import os
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.models import Document
from src.pipeline.baseline import BaselineRAG

def ingest_jsonl_corpus(jsonl_path: str, limit: int = 10000, batch_size: int = 100):
    print(f"Initializing Baseline RAG Pipeline for massive ingestion...")
    rag = BaselineRAG()
    
    print(f"Reading from: {jsonl_path}")
    if limit:
        print(f"Limiting to first {limit} documents for this run to avoid extreme CPU time on local machine.")
        
    count = 0
    batch = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing JSONL", total=limit if limit else None):
            if not line.strip():
                continue
                
            data = json.loads(line)
            doc = Document(
                doc_id=data["doc_id"],
                source=data["source"],
                content=data["content"],
                metadata=data.get("metadata", {})
            )
            batch.append(doc)
            count += 1
            
            if len(batch) >= batch_size:
                rag.ingest_documents(batch)
                batch = []
                
            if limit and count >= limit:
                break
                
    if batch:
        rag.ingest_documents(batch)
        
    print(f"Successfully ingested {count} documents into Qdrant Vector DB.")

if __name__ == "__main__":
    corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'enterprise_corpus.jsonl'))
    
    # Process 10,000 files by default so we can test the evaluation script quickly
    # To run on the full 2.4GB dataset, set limit=None
    ingest_jsonl_corpus(corpus_path, limit=None)
