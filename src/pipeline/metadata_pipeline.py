import os
import sys
import json
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipeline.self_query_parser import SelfQueryParser
from src.pipeline.rerank_pipeline import RerankPipeline
from qdrant_client.models import Filter, FieldCondition, MatchValue

class MetadataPipeline:
    """
    Phase 7: True Pre-Filtering & Self-Query Retrieval.
    Because we successfully patched Qdrant with the 'source' payload,
    we can now pass Qdrant Filters all the way down into our Hybrid pipeline
    so it physically ignores irrelevant documents during the Vector Search!
    """
    def __init__(self, qdrant_path: str = "qdrant_data"):
        self.parser = SelfQueryParser()
        
        # Use our absolute best Phase 5 architecture
        self.rerank_pipeline = RerankPipeline(qdrant_path=qdrant_path, bm25_pkl_path="bm25_semantic_index.pkl")
        
    def search(self, query: str, top_k: int = 10, use_filtering: bool = True) -> List[Dict]:
        
        target_source = None
        qdrant_filter = None
        
        if use_filtering:
            # 1. Parse natural language into JSON filter
            metadata_filters = self.parser.parse_query(query)
            target_source = metadata_filters.get("source")
            if target_source == "null":
                target_source = None
                
            # 2. Build Qdrant Native Filter
            if target_source:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source", 
                            match=MatchValue(value=target_source)
                        )
                    ]
                )
        
        # 3. Native Pre-Filtering! 
        # The Qdrant engine will eliminate bad docs BEFORE Reranking.
        # We also pass source_filter in case Sparse (BM25) needs post-filtering
        results = self.rerank_pipeline.search(
            query, top_k=top_k, fetch_k=60, 
            qdrant_filter=qdrant_filter, source_filter=target_source
        )
        
        return results
