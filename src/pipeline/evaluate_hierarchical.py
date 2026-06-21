import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipeline.hierarchical_pipeline import HierarchicalBaselineRAG
from src.evaluation.dataset import load_benchmark_dataset
from src.evaluation.metrics import calculate_recall_at_k, calculate_mrr, calculate_precision_at_k, calculate_ndcg

def run_hierarchical_evaluation(jsonl_path: str, top_k: int = 10):
    print("Initializing Hierarchical Baseline RAG Pipeline for Evaluation...")
    rag = HierarchicalBaselineRAG()
    
    questions = load_benchmark_dataset(jsonl_path)
    total_q = len(questions)
    
    if total_q == 0:
        print("No questions found.")
        return {}
        
    recall_1_sum = 0.0
    recall_5_sum = 0.0
    recall_10_sum = 0.0
    precision_5_sum = 0.0
    ndcg_10_sum = 0.0
    mrr_sum = 0.0
    
    print(f"Running evaluation over {total_q} questions...")
    for idx, q in enumerate(questions):
        if idx % 10 == 0: print(f"Progress: {idx}/{total_q}")
            
        query_vector = rag.embedder.embed_query(q.question)
        
        # Because chunks are very small (128 tokens), many chunks might belong to the same doc_id!
        # If we only fetch top_k=10, we might only get 2 unique documents.
        # So we fetch a much larger pool (e.g. 50), and deduplicate to find the true top 10 unique documents.
        retrieved_chunks = rag.vector_db.search(query_vector, top_k=50)
        
        unique_doc_ids = []
        for chunk in retrieved_chunks:
            if chunk["doc_id"] not in unique_doc_ids:
                unique_doc_ids.append(chunk["doc_id"])
            if len(unique_doc_ids) == top_k:
                break
                
        # Fill with padding if we didn't find enough unique docs (rare)
        while len(unique_doc_ids) < top_k:
            unique_doc_ids.append("PAD")
            
        recall_1_sum += calculate_recall_at_k(unique_doc_ids, q.expected_doc_ids, 1)
        recall_5_sum += calculate_recall_at_k(unique_doc_ids, q.expected_doc_ids, 5)
        recall_10_sum += calculate_recall_at_k(unique_doc_ids, q.expected_doc_ids, 10)
        precision_5_sum += calculate_precision_at_k(unique_doc_ids, q.expected_doc_ids, 5)
        ndcg_10_sum += calculate_ndcg(unique_doc_ids, q.expected_doc_ids, 10)
        mrr_sum += calculate_mrr(unique_doc_ids, q.expected_doc_ids)
        
    results = {
        "Total Questions": total_q,
        "Recall@1": round(recall_1_sum / total_q, 4),
        "Recall@5": round(recall_5_sum / total_q, 4),
        "Recall@10": round(recall_10_sum / total_q, 4),
        "Precision@5": round(precision_5_sum / total_q, 4),
        "NDCG@10": round(ndcg_10_sum / total_q, 4),
        "MRR": round(mrr_sum / total_q, 4)
    }
    
    print("\n--- Hierarchical Pipeline Evaluation Results ---")
    for k, v in results.items(): print(f"{k}: {v}")
        
    out_file = os.path.join(os.path.dirname(jsonl_path), '..', 'hierarchical_eval_results.json')
    with open(out_file, 'w') as f: json.dump(results, f, indent=4)
    print(f"Results saved to {out_file}")
        
    return results

if __name__ == "__main__":
    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'all_documents', 'questions.jsonl'))
    run_hierarchical_evaluation(benchmark_path)
