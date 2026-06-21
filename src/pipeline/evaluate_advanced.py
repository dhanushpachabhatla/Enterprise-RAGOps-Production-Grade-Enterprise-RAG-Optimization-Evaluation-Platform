import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipeline.advanced import AdvancedRAG
from src.evaluation.dataset import load_benchmark_dataset
from src.evaluation.metrics import calculate_recall_at_k, calculate_mrr, calculate_precision_at_k, calculate_ndcg

def run_advanced_evaluation(jsonl_path: str, top_k: int = 10):
    print("Initializing Advanced RAG Pipeline for Evaluation...")
    rag = AdvancedRAG()
    
    # Critical: Load the serialized BM25 index from disk so we don't have to re-ingest!
    rag.db.load_bm25()
    
    print(f"Loading benchmark questions from {jsonl_path}...")
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
        if idx % 10 == 0:
            print(f"Progress: {idx}/{total_q}")
            
        # 1. Retrieve top_k using the Advanced Hybrid pipeline
        retrieved_chunks = rag.retrieve(q.question, top_k=top_k)
        retrieved_doc_ids = [chunk["doc_id"] for chunk in retrieved_chunks]
        
        # 2. Calculate metrics
        recall_1_sum += calculate_recall_at_k(retrieved_doc_ids, q.expected_doc_ids, 1)
        recall_5_sum += calculate_recall_at_k(retrieved_doc_ids, q.expected_doc_ids, 5)
        recall_10_sum += calculate_recall_at_k(retrieved_doc_ids, q.expected_doc_ids, 10)
        precision_5_sum += calculate_precision_at_k(retrieved_doc_ids, q.expected_doc_ids, 5)
        ndcg_10_sum += calculate_ndcg(retrieved_doc_ids, q.expected_doc_ids, 10)
        mrr_sum += calculate_mrr(retrieved_doc_ids, q.expected_doc_ids)
        
    results = {
        "Total Questions": total_q,
        "Recall@1": round(recall_1_sum / total_q, 4),
        "Recall@5": round(recall_5_sum / total_q, 4),
        "Recall@10": round(recall_10_sum / total_q, 4),
        "Precision@5": round(precision_5_sum / total_q, 4),
        "NDCG@10": round(ndcg_10_sum / total_q, 4),
        "MRR": round(mrr_sum / total_q, 4)
    }
    
    print("\n--- Advanced Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")
        
    # Save to file
    out_file = os.path.join(os.path.dirname(jsonl_path), '..', 'advanced_eval_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {out_file}")
        
    return results

if __name__ == "__main__":
    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'all_documents', 'questions.jsonl'))
    run_advanced_evaluation(benchmark_path)
