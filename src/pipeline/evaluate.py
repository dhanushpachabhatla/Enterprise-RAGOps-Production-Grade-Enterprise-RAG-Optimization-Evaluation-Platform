import os
import json
from typing import List, Dict, Any
from src.pipeline.baseline import BaselineRAG
from src.evaluation.dataset import load_benchmark_dataset
from src.evaluation.metrics import calculate_recall_at_k, calculate_mrr

def run_evaluation(jsonl_path: str, top_k: int = 10) -> Dict[str, Any]:
    print("Initializing Baseline RAG Pipeline for Evaluation...")
    rag = BaselineRAG()
    
    print(f"Loading benchmark questions from {jsonl_path}...")
    questions = load_benchmark_dataset(jsonl_path)
    
    total_q = len(questions)
    if total_q == 0:
        print("No questions found.")
        return {}
        
    recall_1_sum = 0.0
    recall_5_sum = 0.0
    recall_10_sum = 0.0
    mrr_sum = 0.0
    
    print(f"Running evaluation over {total_q} questions...")
    for idx, q in enumerate(questions):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{total_q}")
            
        # 1. Embed query
        query_vector = rag.embedder.embed_query(q.question)
        
        # 2. Retrieve top_k chunks
        retrieved_chunks = rag.vector_db.search(query_vector, top_k=top_k)
        retrieved_doc_ids = [chunk["doc_id"] for chunk in retrieved_chunks]
        
        # 3. Calculate metrics
        recall_1_sum += calculate_recall_at_k(retrieved_doc_ids, q.expected_doc_ids, 1)
        recall_5_sum += calculate_recall_at_k(retrieved_doc_ids, q.expected_doc_ids, 5)
        recall_10_sum += calculate_recall_at_k(retrieved_doc_ids, q.expected_doc_ids, 10)
        mrr_sum += calculate_mrr(retrieved_doc_ids, q.expected_doc_ids)
        
    results = {
        "Total Questions": total_q,
        "Recall@1": round(recall_1_sum / total_q, 4),
        "Recall@5": round(recall_5_sum / total_q, 4),
        "Recall@10": round(recall_10_sum / total_q, 4),
        "MRR": round(mrr_sum / total_q, 4)
    }
    
    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")
        
    # Save to file
    out_file = os.path.join(os.path.dirname(jsonl_path), '..', 'baseline_eval_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {out_file}")
        
    return results

if __name__ == "__main__":
    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'all_documents', 'questions.jsonl'))
    run_evaluation(benchmark_path)
