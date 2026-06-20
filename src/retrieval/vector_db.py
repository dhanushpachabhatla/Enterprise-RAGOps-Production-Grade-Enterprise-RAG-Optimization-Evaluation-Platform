import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any

class VectorDB:
    def __init__(self, path: str = "qdrant_data", collection_name: str = "baseline_rag"):
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: List[Any], embeddings: List[List[float]]):
        """
        Chunks is a list of Chunk objects (from chunking.fixed)
        """
        if not chunks:
            return
            
        points = []
        for chunk, emb in zip(chunks, embeddings):
            # Qdrant requires ids to be integers or valid UUID strings.
            # We hash our deterministic chunk_id into a UUID.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.chunk_id))
            
            points.append(PointStruct(
                id=point_id,
                vector=emb,
                payload={
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "token_count": chunk.token_count
                }
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points
        
        # Return cleanly formatted results
        return [
            {
                "score": hit.score,
                "doc_id": hit.payload["doc_id"],
                "text": hit.payload["text"],
                "metadata": hit.payload["metadata"]
            }
            for hit in results
        ]
