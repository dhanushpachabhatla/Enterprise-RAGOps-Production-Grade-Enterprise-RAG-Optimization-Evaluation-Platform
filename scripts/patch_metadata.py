import json
import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from tqdm import tqdm

def patch_qdrant_metadata():
    print("Patching Qdrant Payload to include 'source' field...")
    qdrant_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'qdrant_data'))
    client = QdrantClient(path=qdrant_path)
    
    corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'golden_subset.jsonl'))
    
    # Build dictionary of doc_id -> source
    doc_sources = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            doc_sources[data["doc_id"]] = data["source"]
            
    print(f"Loaded {len(doc_sources)} documents from JSONL.")
    
    # We group doc_ids by source to do batch updates (much faster!)
    source_to_docs = {}
    for doc_id, source in doc_sources.items():
        if source not in source_to_docs:
            source_to_docs[source] = []
        source_to_docs[source].append(doc_id)
        
    for source, doc_ids in source_to_docs.items():
        print(f"Updating {len(doc_ids)} documents for source '{source}'...")
        
        # We can't pass 10,000 doc_ids in one MatchAny filter easily without hitting limits, 
        # so we batch them in chunks of 500
        batch_size = 500
        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i:i+batch_size]
            
            from qdrant_client.models import MatchAny
            client.set_payload(
                collection_name="semantic_rag",
                payload={"source": source},
                points=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchAny(any=batch)
                        )
                    ]
                )
            )
            
    print("Successfully patched all Qdrant vectors with 'source' metadata!")

if __name__ == "__main__":
    patch_qdrant_metadata()
