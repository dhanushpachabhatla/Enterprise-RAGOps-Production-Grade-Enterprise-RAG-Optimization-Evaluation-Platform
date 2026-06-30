import sys
import os
import json
import time
import pickle
from tqdm import tqdm
from rank_bm25 import BM25Okapi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ingestion.models import Document
from src.chunking.semantic import SemanticChunker

def build_bm25_index(jsonl_path: str, output_pkl: str):
    print("Initializing Semantic Chunker to rebuild corpus boundaries...")
    chunker = SemanticChunker(chunk_size=512)
    
    count = 0
    batch = []
    all_chunks = []
    
    start_time = time.time()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Chunking Documents", total=50000):
            if not line.strip(): continue
            data = json.loads(line)
            batch.append(Document(
                doc_id=data["doc_id"], source=data["source"],
                content=data["content"], metadata=data.get("metadata", {})
            ))
            count += 1
            
            if len(batch) >= 200:
                chunks = chunker.chunk_batch(batch)
                all_chunks.extend(chunks)
                batch = []
                
    if batch:
        chunks = chunker.chunk_batch(batch)
        all_chunks.extend(chunks)
        
    print(f"Generated {len(all_chunks)} semantic chunks. Building BM25 Sparse Index...")
    
    # BM25 requires tokenized text. We use basic whitespace splitting and lowercasing for speed.
    tokenized_corpus = [chunk.text.lower().split(" ") for chunk in all_chunks]
    
    print("Calculating token frequencies across 173,000+ chunks... (This takes a moment)")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # We must save both the BM25 model AND the original chunk payloads,
    # so when BM25 returns an index, we know what chunk text/doc_id it corresponds to!
    payloads = [
        {
            "doc_id": c.doc_id,
            "text": c.text,
            "metadata": c.metadata
        }
        for c in all_chunks
    ]
    
    export_data = {
        "model": bm25,
        "payloads": payloads
    }
    
    print(f"Saving sparse index to {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(export_data, f)
        
    elapsed = time.time() - start_time
    print(f"Successfully built and saved BM25 Index in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'golden_subset.jsonl'))
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'bm25_semantic_index.pkl'))
    build_bm25_index(corpus_path, out_path)
