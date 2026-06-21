from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import uuid

class HybridDB:
    def __init__(self, path: str = "qdrant_data", collection_name: str = "advanced_rag"):
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.bm25 = None
        self.corpus_chunks = []  # Stores chunks for BM25 lookup
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: List[Any], embeddings: List[List[float]]):
        if not chunks:
            return
            
        points = []
        for chunk, emb in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.chunk_id))
            
            points.append(PointStruct(
                id=point_id,
                vector=emb,
                payload={
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            ))
            
            # Store in memory for BM25
            self.corpus_chunks.append({
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": 0.0 # Placeholder
            })
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def build_bm25(self):
        """Builds the in-memory BM25 index from all ingested chunks and saves it."""
        print(f"Building BM25 Sparse Index over {len(self.corpus_chunks)} chunks...")
        tokenized_corpus = [chunk["text"].lower().split() for chunk in self.corpus_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Save to disk for evaluation runs
        import pickle
        import os
        bm25_path = os.path.join("qdrant_data", "bm25_index.pkl")
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                "bm25": self.bm25,
                "corpus_chunks": self.corpus_chunks
            }, f)
        print("BM25 Index Built and Saved.")

    def load_bm25(self):
        """Loads the BM25 index from disk."""
        import pickle
        import os
        bm25_path = os.path.join("qdrant_data", "bm25_index.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.corpus_chunks = data["corpus_chunks"]
            print(f"Loaded BM25 Index with {len(self.corpus_chunks)} chunks.")
        else:
            print("Warning: BM25 index not found on disk. Run ingestion first.")

    def search_dense(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
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

    def search_sparse(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
            
        tokenized_query = query_text.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices
        top_n = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_n:
            chunk = self.corpus_chunks[idx]
            # Create a copy so we don't modify the stored chunk
            results.append({
                "score": scores[idx],
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })
            
        return results
