import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.parser import DatasetReader
import time

def verify():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'all_documents'))
    print(f"Reading from: {base_dir}")
    
    reader = DatasetReader(base_dir)
    
    start_time = time.time()
    
    count = 0
    source_counts = {}
    
    for doc in reader.read_documents():
        count += 1
        source_counts[doc.source] = source_counts.get(doc.source, 0) + 1
        
        # Stop early for verification purposes
        if count >= 10000:
            break
            
    elapsed = time.time() - start_time
    
    print(f"Processed {count} documents in {elapsed:.2f} seconds.")
    print("Source distribution in sample:")
    for src, c in source_counts.items():
        print(f"  - {src}: {c}")

if __name__ == "__main__":
    verify()
