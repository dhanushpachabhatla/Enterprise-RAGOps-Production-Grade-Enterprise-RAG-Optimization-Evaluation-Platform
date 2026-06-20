import os
import json
import random
from tqdm import tqdm

def generate_golden_subset(corpus_path: str, questions_path: str, out_path: str, num_distractors: int = 10000):
    print("Loading target doc_ids from questions benchmark...")
    target_ids = set()
    with open(questions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            # handle if expected_doc_ids is string or list
            doc_ids = data.get("expected_doc_ids", [])
            if isinstance(doc_ids, str):
                target_ids.add(doc_ids)
            else:
                for d in doc_ids:
                    target_ids.add(d)
                    
    print(f"Found {len(target_ids)} target documents required for evaluation.")
    
    print("Scanning massive corpus to extract target docs and random distractors...")
    # First pass: Count total lines to establish sampling rate (approx 512k)
    # We can skip exact counting and just randomly sample with a rough probability,
    # or just collect the first N distractors. Collecting first N is faster.
    
    saved_count = 0
    distractor_count = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc="Generating Subset"):
            if not line.strip(): continue
            
            # Fast O(1) lookup using json.loads (implemented in C, very fast)
            try:
                doc = json.loads(line)
                doc_id = doc.get("doc_id")
            except:
                continue
                
            if doc_id in target_ids:
                fout.write(line)
                saved_count += 1
            else:
                # 15% chance to keep as a distractor until we hit the limit
                # 512k * 0.15 = ~76k (plenty to hit 50k)
                if distractor_count < num_distractors and random.random() < 0.15:
                    fout.write(line)
                    distractor_count += 1
                    saved_count += 1
                    
    print(f"\nDone! Golden subset saved to {out_path}")
    print(f"Total documents extracted: {saved_count} (Targets + {distractor_count} Distractors)")
    print("You can now safely run embed_corpus.py on this lightweight dataset.")

if __name__ == "__main__":
    corpus = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'enterprise_corpus.jsonl'))
    questions = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'all_documents', 'questions.jsonl'))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'golden_subset.jsonl'))
    
    generate_golden_subset(corpus, questions, out, num_distractors=50000)
