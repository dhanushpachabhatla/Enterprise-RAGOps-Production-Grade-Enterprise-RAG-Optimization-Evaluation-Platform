import os
import sys
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipeline.hybrid_pipeline import HybridRetrievalPipeline

class RerankPipeline:
    """
    Phase 5: Cross-Encoder Re-ranking Pipeline.
    Uses Python Hybrid Pipeline (alpha=0.5) to fetch top 30 unique documents, 
    then uses a Cross-Encoder to re-rank them based on deep semantic relevance.
    """
    def __init__(self, qdrant_path: str = "qdrant_data", bm25_pkl_path: str = "bm25_semantic_index.pkl"):
        # 1. Initialize our base Python Hybrid pipeline
        self.hybrid_pipeline = HybridRetrievalPipeline(qdrant_path=qdrant_path, bm25_pkl_path=bm25_pkl_path)
        
        # 2. Initialize the Cross-Encoder Re-ranker
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Cross-Encoder (BAAI/bge-reranker-base) on {device.upper()}...")
        self.cross_encoder = CrossEncoder("BAAI/bge-reranker-base", device=device)
        
    def search(self, query: str, top_k: int = 10, fetch_k: int = 60, alpha: float = 0.5) -> List[Dict]:
        """
        1. Fetch top 30 from Hybrid Pipeline (with alpha=0.5 to maximize Recall)
        2. Score each chunk against the query using CrossEncoder.
        3. Sort and return top 10.
        """
        # Step 1: Hybrid Retrieval. We ask for 30 unique documents so the reranker has a large pool to choose from.
        # We keep fetch_k at 60 (pulling 60 from Dense, 60 from Sparse to merge).
        hybrid_results = self.hybrid_pipeline.search(query, top_k=30, fetch_k=fetch_k, alpha=alpha)
        
        if not hybrid_results:
            return []
            
        # Step 2: Prepare pairs for Cross-Encoder
        # The Cross-Encoder takes a list of pairs: [[query, text1], [query, text2], ...]
        pairs = [[query, chunk["text"]] for chunk in hybrid_results]
        
        # Step 3: Get Re-ranking scores
        scores = self.cross_encoder.predict(pairs)
        
        # Step 4: Attach scores to the chunks and sort
        for chunk, score in zip(hybrid_results, scores):
            chunk["rerank_score"] = float(score)
            
        # Sort by rerank_score descending
        reranked_results = sorted(hybrid_results, key=lambda x: x["rerank_score"], reverse=True)
        
        # Step 5: Return the top_k (usually 10)
        return reranked_results[:top_k]
