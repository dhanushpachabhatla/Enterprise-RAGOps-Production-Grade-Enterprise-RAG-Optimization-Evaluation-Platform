import sys
import os
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.models import Document
from src.pipeline.advanced import AdvancedRAG

def ingest_advanced(jsonl_path: str, limit: int = None, batch_size: int = 100):
    import time
    import torch
    
    print("=== Hardware Check ===")
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)} (CUDA: True)")
    else:
        print("GPU Detected: None (Using CPU)")
    print("======================\n")

    print(f"Initializing Advanced RAG Pipeline (Semantic Chunking + Hybrid BM25)")
    start_time = time.time()
    rag = AdvancedRAG()
    
    count = 0
    batch = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing JSONL", total=limit if limit else None):
            if not line.strip(): continue
                
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
        
    print("Finalizing Hybrid Ingestion (Building BM25 Index)...")
    rag.finalize_ingestion()
        
    elapsed = time.time() - start_time
    print(f"Successfully ingested {count} documents into Hybrid DB.")
    print(f"Total time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")

if __name__ == "__main__":
    corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'golden_subset.jsonl'))
    ingest_advanced(corpus_path, limit=None)
