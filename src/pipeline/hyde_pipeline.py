import os
import sys
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import List, Dict
from src.pipeline.query_rewriter import QueryRewriter
from src.embeddings.embedder import BGEEmbedder
from src.retrieval.vector_db import VectorDB

class HydePipeline:
    """
    Phase 6.1: Context Engineering (HyDE Dense Only).
    Intercepts the user's query, generates a Hypothetical Document, 
    and passes the combined text EXCLUSIVELY to the Dense Vector Pipeline.
    """
    def __init__(self, qdrant_path: str = "qdrant_data", bm25_pkl_path: str = None):
        self.rewriter = QueryRewriter()
        
        # Pure Dense vector components to avoid BM25 interference
        self.embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        self.vector_db = VectorDB(collection_name="semantic_rag", path=qdrant_path)
        
    def search(self, query: str, top_k: int = 10, fetch_k: int = 60, alpha: float = 0.5) -> List[Dict]:
        # 1. Generate Hypothetical Document (instant because of our JSON cache!)
        hyde_doc = self.rewriter.generate_hyde_document(query)
        
        # 2. Combine them
        enhanced_query = f"{query}\n\n{hyde_doc}"
        
        # 3. Embed the enhanced query
        query_emb = self.embedder.embed_query(enhanced_query)
        
        # 4. Search EXCLUSIVELY in the Dense Vector DB
        return self.vector_db.search(query_emb, top_k=top_k)
