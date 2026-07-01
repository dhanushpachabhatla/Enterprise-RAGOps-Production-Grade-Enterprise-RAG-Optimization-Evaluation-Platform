import os
import sys
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipeline.self_query_parser import SelfQueryParser
from src.embeddings.embedder import BGEEmbedder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

class MetadataPipeline:
    """
    Phase 7: Metadata Filtering & Self-Query Retrieval.
    Parses the user query to extract strict JSON metadata filters,
    and applies them directly to Qdrant's search engine to prune the database.
    """
    def __init__(self, qdrant_path: str = "qdrant_data"):
        self.parser = SelfQueryParser()
        self.embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        
        # We need direct access to QdrantClient to build complex Filter objects
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = "semantic_rag"
        
    def search(self, query: str, top_k: int = 10, use_filtering: bool = True) -> List[Dict]:
        print(f"\n[Metadata] Query: {query}")
        
        qdrant_filter = None
        if use_filtering:
            # 1. Parse natural language into JSON filter
            metadata_filters = self.parser.parse_query(query)
            print(f"[Metadata] Extracted Filters: {metadata_filters}")
            
            # 2. Convert JSON dictionary into Qdrant Filter Object
            source = metadata_filters.get("source")
            if source:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            # We stored the source under `metadata.source` in the payload during ingestion
                            key="metadata.source", 
                            match=MatchValue(value=source)
                        )
                    ]
                )
                print(f"[Metadata] Applied strict Vector DB filter: source == '{source}'")
            else:
                print("[Metadata] No filters extracted. Running open vector search.")
        
        # 3. Embed Query
        query_emb = self.embedder.embed_query(query)
        
        # 4. Filtered Vector Search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb,
            query_filter=qdrant_filter, # The magic happens here!
            limit=top_k
        ).points
        
        return [
            {
                "score": hit.score,
                "doc_id": hit.payload["doc_id"],
                "text": hit.payload["text"],
                "metadata": hit.payload["metadata"]
            }
            for hit in results
        ]
