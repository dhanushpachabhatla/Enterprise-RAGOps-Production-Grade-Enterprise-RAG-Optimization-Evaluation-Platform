import sys
import os
import json
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.parser import DatasetReader

def convert_to_jsonl(input_dir: str, output_file: str):
    print(f"Reading from: {input_dir}")
    print(f"Writing to: {output_file}")
    
    reader = DatasetReader(input_dir)
    
    # We use a progress bar, though we don't know the exact total upfront unless we pre-count.
    # From earlier analysis, we know there are about 511,962 files.
    total_files = 511962 
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(reader.read_documents(), total=total_files, desc="Converting Documents"):
            # doc.model_dump_json() serializes the Pydantic model to a JSON string
            f.write(doc.model_dump_json() + "\n")
            
    print(f"Successfully converted documents to {output_file}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'all_documents'))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'enterprise_corpus.jsonl'))
    
    convert_to_jsonl(base_dir, output_path)
