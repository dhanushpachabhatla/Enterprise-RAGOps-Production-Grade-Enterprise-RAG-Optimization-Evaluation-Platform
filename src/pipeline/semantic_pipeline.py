from typing import List
from src.chunking.semantic import SemanticChunker
from src.embeddings.embedder import BGEEmbedder
from src.retrieval.vector_db import VectorDB
from src.ingestion.models import Document

class SemanticBaselineRAG:
    """
    Experiment 1A: Ablation Study Pipeline.
    Isolates Semantic Chunking while keeping Dense Retrieval and Ranking identical to Baseline.
    """
    def __init__(self):
        # The ONLY change: Swap FixedChunker for SemanticChunker
        self.chunker = SemanticChunker(chunk_size=512)
        
        # Keep everything else identical to Baseline
        self.embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        self.vector_db = VectorDB(collection_name="semantic_rag")

    def ingest_documents(self, documents: List[Document]):
        # Spacy multi-threading optimization: chunk all documents at once
        all_chunks = self.chunker.chunk_batch(documents)

        if not all_chunks:
            return

        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_db.upsert_chunks(all_chunks, embeddings)
