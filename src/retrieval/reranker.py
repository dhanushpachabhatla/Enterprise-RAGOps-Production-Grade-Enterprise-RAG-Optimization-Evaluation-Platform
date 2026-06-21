from typing import List, Dict, Any

class RRFMerger:
    def __init__(self, k: int = 60):
        """
        k is a constant used in RRF. 60 is the industry standard default.
        """
        self.k = k

    def merge_results(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Merges two lists of retrieved documents using Reciprocal Rank Fusion (RRF).
        RRF Score = 1 / (k + rank)
        """
        doc_scores = {}
        doc_store = {}
        
        # Note: We rank by doc_id to avoid returning duplicate chunks from the same document.
        # If multiple chunks from the same document are retrieved, we take the highest rank.
        
        # Process Dense
        for rank, hit in enumerate(dense_results):
            doc_id = hit["doc_id"]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
                doc_store[doc_id] = hit
            doc_scores[doc_id] += 1.0 / (self.k + rank + 1)
            
        # Process Sparse (BM25)
        for rank, hit in enumerate(sparse_results):
            doc_id = hit["doc_id"]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
                doc_store[doc_id] = hit
            doc_scores[doc_id] += 1.0 / (self.k + rank + 1)
            
        # Sort by RRF score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format top_k output
        merged = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = doc_store[doc_id].copy()
            doc["score"] = score  # Overwrite Qdrant/BM25 score with unified RRF score
            merged.append(doc)
            
        return merged
