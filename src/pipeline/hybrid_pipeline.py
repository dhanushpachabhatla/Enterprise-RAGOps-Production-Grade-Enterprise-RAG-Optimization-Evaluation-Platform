import os
import pickle
import numpy as np
import hashlib
from typing import List, Dict, Any
from src.embeddings.embedder import BGEEmbedder
from src.retrieval.vector_db import VectorDB

class HybridRetrievalPipeline:
    """
    Phase 4: Hybrid Retrieval Pipeline.
    Combines Qdrant Dense Vector Search with BM25 Sparse Token Search using Reciprocal Rank Fusion.
    """
    def __init__(self, qdrant_path: str = "qdrant_data", bm25_pkl_path: str = "bm25_semantic_index.pkl"):
        # 1. Initialize Dense Retriever (Qdrant Semantic RAG)
        print("Initializing Dense Retriever (BAAI/bge-small-en-v1.5 + Qdrant)...")
        self.embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        self.vector_db = VectorDB(path=qdrant_path, collection_name="semantic_rag", in_memory=False)
        
        # 2. Initialize Sparse Retriever (BM25)
        print(f"Loading BM25 Sparse Index from {bm25_pkl_path}...")
        if not os.path.exists(bm25_pkl_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_pkl_path}. Run bm25_index.py first!")
            
        with open(bm25_pkl_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25_model = data["model"]
            self.bm25_payloads = data["payloads"]
            
        # 3. Load doc_id -> source map for Sparse Post-Filtering
        import json
        self.doc_sources = {}
        corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'golden_subset.jsonl'))
        if os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    self.doc_sources[data["doc_id"]] = data["source"]
            
    def _rrf(self, dense_results: List[Dict], sparse_results: List[Dict], k: int = 60, alpha: float = 0.8) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) with weights.
        alpha controls the weight of the Dense vector (0.0 to 1.0).
        """
        rrf_scores = {}
        payload_map = {}
        
        def get_key(res):
            return hashlib.md5(res["text"].encode("utf-8")).hexdigest()
            
        # Process Dense
        for rank, res in enumerate(dense_results):
            key = get_key(res)
            payload_map[key] = res
            rrf_scores[key] = rrf_scores.get(key, 0.0) + alpha * (1.0 / (k + rank + 1))
            
        # Process Sparse
        for rank, res in enumerate(sparse_results):
            key = get_key(res)
            if key not in payload_map:
                payload_map[key] = res
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 - alpha) * (1.0 / (k + rank + 1))
            
        # Sort chunks by their combined RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        return [payload_map[k] for k in sorted_keys]

    def search(self, query: str, top_k: int = 10, fetch_k: int = 60, alpha: float = 0.8, qdrant_filter = None, source_filter: str = None) -> List[Dict]:
        """
        Fetches `fetch_k` from both dense and sparse, fuses them, and deduplicates to `top_k`.
        Applies True Pre-Filtering to Dense Search (Qdrant), and Post-Filtering to Sparse Search (BM25).
        """
        # 1. Dense Search (True Pre-Filtering at the Database Level!)
        query_vector = self.embedder.embed_query(query)
        dense_results = self.vector_db.search(query_vector, top_k=fetch_k, query_filter=qdrant_filter)
        
        # 2. Sparse Search (BM25 does not support native pre-filtering, so we Post-Filter here)
        tokenized_query = query.lower().split(" ")
        sparse_scores = self.bm25_model.get_scores(tokenized_query)
        top_sparse_indices = np.argsort(sparse_scores)[::-1]
        
        sparse_results = []
        for idx in top_sparse_indices:
            if len(sparse_results) >= fetch_k:
                break
            if sparse_scores[idx] > 0:
                payload = self.bm25_payloads[idx]
                
                # Apply Sparse Post-Filtering if a source filter is provided
                if source_filter:
                    doc_source = self.doc_sources.get(payload["doc_id"])
                    if doc_source != source_filter:
                        continue # Skip this chunk!
                
                sparse_results.append(payload)
                
        # 3. Fuse via RRF
        fused_chunks = self._rrf(dense_results, sparse_results, k=60, alpha=alpha)
        
        # 4. Deduplicate by doc_id to get top_k unique documents
        unique_doc_ids = []
        final_results = []
        
        for chunk in fused_chunks:
            if chunk["doc_id"] not in unique_doc_ids:
                unique_doc_ids.append(chunk["doc_id"])
                final_results.append(chunk)
            if len(unique_doc_ids) == top_k:
                break
                
        return final_results
