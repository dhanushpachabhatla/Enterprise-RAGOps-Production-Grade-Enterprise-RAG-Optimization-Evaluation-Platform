from sentence_transformers import SentenceTransformer
from typing import List

class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        # We load a local model for efficient embedding
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Takes a list of strings and returns their dense vector representations.
        For BGE models, standard queries might need instructions, but for documents we just embed.
        """
        if not texts:
            return []
        
        # Output is typically a numpy array, convert to list of floats for qdrant
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
        
    def embed_query(self, query: str) -> List[float]:
        """
        BGE models often perform better when the query is prefixed.
        """
        instruction = "Represent this sentence for searching relevant passages: "
        emb = self.model.encode([instruction + query], normalize_embeddings=True)
        return emb[0].tolist()
