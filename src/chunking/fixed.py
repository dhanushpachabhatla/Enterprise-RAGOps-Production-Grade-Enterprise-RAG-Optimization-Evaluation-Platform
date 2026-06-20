import tiktoken
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FixedTokenChunker:
    def __init__(self, model_name: str = "gpt-3.5-turbo", chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        if not text:
            return []
            
        metadata = metadata or {}
        # Disable special token checks so tiktoken doesn't crash on <|endoftext|> strings in the corpus
        tokens = self.tokenizer.encode(text, disallowed_special=())
        
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                doc_id=doc_id,
                text=chunk_text,
                token_count=len(chunk_tokens),
                metadata=metadata
            ))
            
            chunk_idx += 1
            start += self.chunk_size - self.overlap
            
        return chunks
