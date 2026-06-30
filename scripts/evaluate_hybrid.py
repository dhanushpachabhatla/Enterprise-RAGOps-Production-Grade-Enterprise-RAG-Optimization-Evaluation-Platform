import os
import sys
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.hybrid_pipeline import HybridRetrievalPipeline
from src.evaluation.dataset import load_benchmark_dataset
from src.evaluation.metrics import calculate_recall_at_k, calculate_mrr, calculate_precision_at_k, calculate_ndcg

def run_evaluation():
    print("Initializing Hybrid Retrieval Pipeline (Qdrant Dense + BM25 Sparse)...")
    
    qdrant_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'qdrant_data'))
    bm25_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bm25_semantic_index.pkl'))
    
    try:
        pipeline = HybridRetrievalPipeline(qdrant_path=qdrant_path, bm25_pkl_path=bm25_path)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run 'python src/retrieval/bm25_index.py' first!")
        return
        
    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'all_documents', 'questions.jsonl'))
    questions = load_benchmark_dataset(benchmark_path)
    total_q = len(questions)
    
    recall_1_sum = recall_5_sum = recall_10_sum = 0.0
    precision_5_sum = ndcg_10_sum = mrr_sum = 0.0
    
    print(f"\nEvaluating Hybrid Pipeline across {total_q} queries...")
    for idx, q in enumerate(tqdm(questions, desc="Evaluating")):
        # The pipeline handles RRF fusion and deduplication automatically!
        retrieved_docs = pipeline.search(q.question, top_k=10, fetch_k=60, alpha=0.8)
        
        # We already have the top 10 unique documents
        unique_doc_ids = [doc["doc_id"] for doc in retrieved_docs]
        
        # Pad if less than 10 (rare)
        while len(unique_doc_ids) < 10:
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
    
    print("\n--- Hybrid BM25 Pipeline Evaluation Results ---")
    for k, v in results.items(): print(f"{k}: {v}")
        
    out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hybrid_eval_results.json'))
    with open(out_file, 'w') as f: json.dump(results, f, indent=4)
    print(f"\nSuccessfully saved results to {out_file}")

if __name__ == "__main__":
    run_evaluation()
