import os
import sys
import json
import time
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.hyde_pipeline import HydePipeline
from src.evaluation.dataset import load_benchmark_dataset
from src.evaluation.metrics import calculate_recall_at_k, calculate_mrr, calculate_precision_at_k, calculate_ndcg

def run_evaluation():
    print("Initializing HyDE Evaluation Pipeline (HyDE -> Hybrid -> Cross-Encoder)...")
    
    qdrant_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'qdrant_data'))
    bm25_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bm25_semantic_index.pkl'))
    
    try:
        pipeline = HydePipeline(qdrant_path=qdrant_path, bm25_pkl_path=bm25_path)
    except Exception as e:
        print(f"\nERROR: {e}")
        return
        
    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'all_documents', 'questions.jsonl'))
    questions = load_benchmark_dataset(benchmark_path)
    total_q = len(questions)
    
    recall_1_sum = recall_5_sum = recall_10_sum = 0.0
    precision_5_sum = ndcg_10_sum = mrr_sum = 0.0
    
    print(f"\nEvaluating HyDE Pipeline across {total_q} queries...")
    start_time = time.time()
    
    for idx, q in enumerate(tqdm(questions, desc="Evaluating")):
        # We fetch 60 (to maximize raw document pool), fuse with RRF, then Rerank to top 10.
        retrieved_docs = pipeline.search(q.question, top_k=10, fetch_k=60, alpha=0.5)
        
        unique_doc_ids = [doc["doc_id"] for doc in retrieved_docs]
        
        while len(unique_doc_ids) < 10:
            unique_doc_ids.append("PAD")
            
        recall_1_sum += calculate_recall_at_k(unique_doc_ids, q.expected_doc_ids, 1)
        recall_5_sum += calculate_recall_at_k(unique_doc_ids, q.expected_doc_ids, 5)
        recall_10_sum += calculate_recall_at_k(unique_doc_ids, q.expected_doc_ids, 10)
        precision_5_sum += calculate_precision_at_k(unique_doc_ids, q.expected_doc_ids, 5)
        ndcg_10_sum += calculate_ndcg(unique_doc_ids, q.expected_doc_ids, 10)
        mrr_sum += calculate_mrr(unique_doc_ids, q.expected_doc_ids)
        
    elapsed = time.time() - start_time
        
    results = {
        "Total Questions": total_q,
        "Recall@1": round(recall_1_sum / total_q, 4),
        "Recall@5": round(recall_5_sum / total_q, 4),
        "Recall@10": round(recall_10_sum / total_q, 4),
        "Precision@5": round(precision_5_sum / total_q, 4),
        "NDCG@10": round(ndcg_10_sum / total_q, 4),
        "MRR": round(mrr_sum / total_q, 4),
        "Total_Time_Seconds": round(elapsed, 2)
    }
    
    print("\n--- Phase 6: HyDE Pipeline Evaluation Results ---")
    for k, v in results.items(): print(f"{k}: {v}")
        
    out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hyde_eval_results.json'))
    with open(out_file, 'w') as f: json.dump(results, f, indent=4)
    print(f"\nSuccessfully saved results to {out_file}")

if __name__ == "__main__":
    run_evaluation()
