from typing import List, Dict, Any
from src.chunking.semantic import SemanticChunker
from src.embeddings.embedder import BGEEmbedder
from src.retrieval.hybrid_db import HybridDB
from src.retrieval.reranker import RRFMerger
from src.ingestion.models import Document

class AdvancedRAG:
    def __init__(self):
        # 1. Semantic Chunking (respects sentence boundaries)
        self.chunker = SemanticChunker(chunk_size=512)
        
        # 2. Dense Embedder (using identical model as baseline for fair comparison)
        self.embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        
        # 3. Hybrid Database (Dense + Sparse BM25)
        self.db = HybridDB(collection_name="advanced_rag")
        
        # 4. RRF Merger
        self.reranker = RRFMerger(k=60)

    def ingest_documents(self, documents: List[Document]):
        """Chunks, embeds, and stores documents in Qdrant."""
        # Spacy multi-threading optimization: chunk all documents at once
        all_chunks = self.chunker.chunk_batch(documents)

        if not all_chunks:
            return

        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.db.upsert_chunks(all_chunks, embeddings)

    def finalize_ingestion(self):
        """Must be called after all documents are ingested to build the in-memory BM25 index."""
        self.db.build_bm25()

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Executes Hybrid Search and returns RRF re-ranked results."""
        # 1. Dense Search
        query_vector = self.embedder.embed_query(query)
        # Fetch 2x top_k to give the reranker more options
        dense_results = self.db.search_dense(query_vector, top_k=top_k * 2)
        
        # 2. Sparse Search (BM25)
        sparse_results = self.db.search_sparse(query, top_k=top_k * 2)
        
        # 3. Hybrid RRF Merge
        merged_results = self.reranker.merge_results(dense_results, sparse_results, top_k=top_k)
        return merged_results
