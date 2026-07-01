import os
import sys
import json
import time
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.metadata_pipeline import MetadataPipeline
from src.evaluation.dataset import load_benchmark_dataset
from src.evaluation.metrics import calculate_recall_at_k, calculate_mrr, calculate_precision_at_k, calculate_ndcg

def run_evaluation_pass(pipeline, questions, use_filtering: bool, pass_name: str):
    total_q = len(questions)
    recall_1_sum = recall_5_sum = recall_10_sum = 0.0
    precision_5_sum = ndcg_10_sum = mrr_sum = 0.0
    
    print(f"\n--- Running {pass_name} Pass ---")
    start_time = time.time()
    
    for idx, q in enumerate(tqdm(questions, desc=f"Evaluating ({pass_name})")):
        retrieved_docs = pipeline.search(q.question, top_k=10, use_filtering=use_filtering)
        
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
        
    return {
        "Total Questions": total_q,
        "Recall@1": round(recall_1_sum / total_q, 4),
        "Recall@5": round(recall_5_sum / total_q, 4),
        "Recall@10": round(recall_10_sum / total_q, 4),
        "Precision@5": round(precision_5_sum / total_q, 4),
        "NDCG@10": round(ndcg_10_sum / total_q, 4),
        "MRR": round(mrr_sum / total_q, 4),
        "Total_Time_Seconds": round(elapsed, 2)
    }

def run_evaluation():
    print("Initializing Metadata Filtering & Self-Query Pipeline...")
    
    qdrant_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'qdrant_data'))
    
    try:
        pipeline = MetadataPipeline(qdrant_path=qdrant_path)
    except Exception as e:
        print(f"\nERROR: {e}")
        return
        
    # CRITICAL: We use `extra_questions.jsonl` which explicitly tests metadata
    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'extra_questions.jsonl'))
    questions = load_benchmark_dataset(benchmark_path)
    
    # Run 1: Unfiltered (Baseline Dense Vector Search)
    unfiltered_results = run_evaluation_pass(pipeline, questions, use_filtering=False, pass_name="UNFILTERED (Baseline)")
    
    # Run 2: Filtered (Self-Query Metadata Extraction)
    filtered_results = run_evaluation_pass(pipeline, questions, use_filtering=True, pass_name="FILTERED (Self-Query)")
    
    print("\n=======================================================")
    print("      PHASE 7: METADATA FILTERING RESULTS COMPARISON     ")
    print("=======================================================")
    print(f"{'Metric':<15} | {'Unfiltered (Baseline)':<22} | {'Filtered (Self-Query)'}")
    print("-" * 65)
    for k in unfiltered_results.keys():
        print(f"{k:<15} | {unfiltered_results[k]:<22} | {filtered_results[k]}")
        
    out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'metadata_eval_results.json'))
    with open(out_file, 'w') as f: 
        json.dump({"unfiltered": unfiltered_results, "filtered": filtered_results}, f, indent=4)
    print(f"\nSuccessfully saved comparative results to {out_file}")

if __name__ == "__main__":
    run_evaluation()
