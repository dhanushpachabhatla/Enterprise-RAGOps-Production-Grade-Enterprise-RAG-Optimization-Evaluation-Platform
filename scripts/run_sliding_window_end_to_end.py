import sys
import os
import json
import time
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.models import Document
from src.pipeline.sliding_window_pipeline import SlidingWindowBaselineRAG
from src.evaluation.dataset import load_benchmark_dataset
from src.evaluation.metrics import calculate_recall_at_k, calculate_mrr, calculate_precision_at_k, calculate_ndcg

def run_end_to_end():
    print("=== Hardware Check ===")
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)} (CUDA: True)")
    else:
        print("GPU Detected: None (Using CPU)")
    print("======================\n")

    print("Initializing Sliding Window RAG Pipeline (IN-MEMORY MODE)...")
    start_time = time.time()
    
    rag = SlidingWindowBaselineRAG(in_memory=True)
    
    corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'golden_subset.jsonl'))
    batch_size = 100
    count = 0
    batch = []
    
    print("\n--- PHASE 1: RAM INGESTION ---")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Ingesting to RAM", total=50000):
            if not line.strip(): continue
            data = json.loads(line)
            batch.append(Document(
                doc_id=data["doc_id"], source=data["source"],
                content=data["content"], metadata=data.get("metadata", {})
            ))
            count += 1
            
            if len(batch) >= batch_size:
                rag.ingest_documents(batch)
                batch = []
                
    if batch:
        rag.ingest_documents(batch)
        
    elapsed = time.time() - start_time
    print(f"Successfully ingested {count} documents into RAM.")
    print(f"Ingestion time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
    
    # ---------------- EVALUATION PHASE ----------------
    print("\n--- PHASE 2: EVALUATION AGAINST RAM DB ---")
    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'all_documents', 'questions.jsonl'))
    questions = load_benchmark_dataset(benchmark_path)
    total_q = len(questions)
    
    recall_1_sum = recall_5_sum = recall_10_sum = 0.0
    precision_5_sum = ndcg_10_sum = mrr_sum = 0.0
    
    for idx, q in enumerate(questions):
        if idx % 10 == 0: print(f"Evaluating Progress: {idx}/{total_q}")
            
        query_vector = rag.embedder.embed_query(q.question)
        
        # Deduplication logic (highly important for 256-token overlap as facts are duplicated across chunks)
        retrieved_chunks = rag.vector_db.search(query_vector, top_k=50)
        
        unique_doc_ids = []
        for chunk in retrieved_chunks:
            if chunk["doc_id"] not in unique_doc_ids:
                unique_doc_ids.append(chunk["doc_id"])
            if len(unique_doc_ids) == 10: break
                
        while len(unique_doc_ids) < 10: unique_doc_ids.append("PAD")
            
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
    
    print("\n--- Sliding Window Pipeline Evaluation Results ---")
    for k, v in results.items(): print(f"{k}: {v}")
        
    out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sliding_window_eval_results.json'))
    with open(out_file, 'w') as f: json.dump(results, f, indent=4)
    print(f"\nSuccessfully saved mathematical results to {out_file}")
    print("Database is now being wiped from RAM safely. Goodbye!")

if __name__ == "__main__":
    run_end_to_end()
