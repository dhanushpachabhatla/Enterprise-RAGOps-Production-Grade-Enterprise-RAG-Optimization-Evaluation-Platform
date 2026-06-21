from typing import List
from src.chunking.hierarchical import HierarchicalChunker
from src.embeddings.embedder import BGEEmbedder
from src.retrieval.vector_db import VectorDB
from src.ingestion.models import Document

class HierarchicalBaselineRAG:
    """
    Experiment 1B: Ablation Study Pipeline.
    Isolates Hierarchical Chunking while keeping Dense Retrieval identical to Baseline.
    """
    def __init__(self, in_memory: bool = False):
        # The ONLY change: Swap to HierarchicalChunker
        self.chunker = HierarchicalChunker(parent_size=512, child_size=128)
        
        # Keep everything else identical to Baseline
        self.embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        self.vector_db = VectorDB(collection_name="hierarchical_rag", in_memory=in_memory)

    def ingest_documents(self, documents: List[Document]):
        all_child_chunks = self.chunker.chunk_batch(documents)

        if not all_child_chunks:
            return

        texts = [c.text for c in all_child_chunks]
        # Chunks are 4x smaller (128 tokens), so we can push 4x more through the GPU at once!
        embeddings = self.embedder.embed_texts(texts, batch_size=512)
        self.vector_db.upsert_chunks(all_child_chunks, embeddings)
