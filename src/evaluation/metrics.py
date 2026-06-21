from typing import List

def calculate_recall_at_k(retrieved_doc_ids: List[str], expected_doc_ids: List[str], k: int) -> float:
    """
    Calculates Recall@K. 
    1.0 if ANY of the expected_doc_ids appear in the top K retrieved_doc_ids, else 0.0.
    For dataset aggregates, we average this binary score.
    """
    if not expected_doc_ids:
        return 0.0
        
    top_k_retrieved = retrieved_doc_ids[:k]
    
    for expected_id in expected_doc_ids:
        if expected_id in top_k_retrieved:
            return 1.0
            
    return 0.0

def calculate_mrr(retrieved_doc_ids: List[str], expected_doc_ids: List[str]) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR).
    Finds the highest rank among the retrieved_doc_ids that matches ANY expected_doc_ids.
    Returns 1 / rank (1-indexed). Returns 0.0 if none found.
    """
    if not expected_doc_ids:
        return 0.0
        
    for idx, retrieved_id in enumerate(retrieved_doc_ids):
        if retrieved_id in expected_doc_ids:
            return 1.0 / (idx + 1)
            
    return 0.0

def calculate_precision_at_k(retrieved_doc_ids: List[str], expected_doc_ids: List[str], k: int) -> float:
    """
    Calculates Precision@K.
    (Number of relevant documents in top K) / K.
    """
    if not expected_doc_ids or k == 0:
        return 0.0
        
    top_k_retrieved = retrieved_doc_ids[:k]
    relevant_count = sum(1 for doc_id in top_k_retrieved if doc_id in expected_doc_ids)
    return relevant_count / k

def calculate_ndcg(retrieved_doc_ids: List[str], expected_doc_ids: List[str], k: int = 10) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG) assuming binary relevance.
    """
    import math
    if not expected_doc_ids:
        return 0.0
        
    top_k_retrieved = retrieved_doc_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k_retrieved):
        if doc_id in expected_doc_ids:
            dcg += 1.0 / math.log2(i + 2)  # i is 0-indexed
            
    idcg = 0.0
    num_relevant = min(len(expected_doc_ids), k)
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
