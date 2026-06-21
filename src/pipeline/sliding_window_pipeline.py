from typing import List
from src.chunking.semantic import SemanticChunker
from src.embeddings.embedder import BGEEmbedder
from src.retrieval.vector_db import VectorDB
from src.ingestion.models import Document

class SlidingWindowBaselineRAG:
    """
    Experiment 1C: Sliding Window Chunking.
    Uses Semantic boundaries but enforces a massive 256-token overlap.
    """
    def __init__(self, in_memory: bool = False):
        self.chunker = SemanticChunker(chunk_size=512, overlap=256)
        
        self.embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        self.vector_db = VectorDB(collection_name="sliding_window_rag", in_memory=in_memory)

    def ingest_documents(self, documents: List[Document]):
        all_chunks = self.chunker.chunk_batch(documents)

        if not all_chunks:
            return

        texts = [c.text for c in all_chunks]
        # Since chunks are large again (512), we drop batch_size back to 128
        embeddings = self.embedder.embed_texts(texts, batch_size=128)
        self.vector_db.upsert_chunks(all_chunks, embeddings)
