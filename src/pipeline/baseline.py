from typing import List, Dict, Any
from src.chunking.fixed import FixedTokenChunker
from src.embeddings.embedder import BGEEmbedder
from src.retrieval.vector_db import VectorDB
from src.generation.llm import BaselineLLM
from src.ingestion.models import Document

class BaselineRAG:
    def __init__(self, 
                 chunker: FixedTokenChunker = None, 
                 embedder: BGEEmbedder = None, 
                 vector_db: VectorDB = None, 
                 llm: BaselineLLM = None):
        self.chunker = chunker or FixedTokenChunker()
        self.embedder = embedder or BGEEmbedder()
        self.vector_db = vector_db or VectorDB()
        self.llm = llm or BaselineLLM()

    def ingest_documents(self, documents: List[Document]):
        """
        Process a batch of Document objects:
        1. Chunk them
        2. Embed them
        3. Store in Qdrant
        """
        all_chunks = []
        
        print(f"Chunking {len(documents)} documents...")
        for doc in documents:
            chunks = self.chunker.chunk_document(
                doc_id=doc.doc_id, 
                text=doc.content, 
                metadata=doc.metadata
            )
            all_chunks.extend(chunks)
            
        if not all_chunks:
            print("No chunks generated.")
            return

        print(f"Embedding {len(all_chunks)} chunks...")
        texts_to_embed = [c.text for c in all_chunks]
        embeddings = self.embedder.embed_texts(texts_to_embed)

        print(f"Upserting to Vector DB...")
        self.vector_db.upsert_chunks(all_chunks, embeddings)
        print("Done.")

    def ask(self, query: str, top_k: int = 10) -> str:
        """
        1. Embed query
        2. Retrieve top_k chunks from Qdrant
        3. Generate answer via LLM
        """
        query_vector = self.embedder.embed_query(query)
        retrieved_chunks = self.vector_db.search(query_vector, top_k=top_k)
        
        if not retrieved_chunks:
            return "No relevant documents found."
            
        answer = self.llm.generate_answer(query, retrieved_chunks)
        return answer
