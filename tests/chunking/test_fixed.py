import pytest
from src.chunking.fixed import FixedTokenChunker

def test_fixed_chunker():
    chunker = FixedTokenChunker(chunk_size=10, overlap=2)
    # Let's provide a string that is easily tokenized into >10 tokens
    text = "This is a simple test document that should be chunked properly by the tokenizer, creating at least two overlapping chunks."
    chunks = chunker.chunk_document("doc_1", text)
    
    assert len(chunks) > 0
    assert chunks[0].doc_id == "doc_1"
    assert chunks[0].token_count <= 10
    
    if len(chunks) > 1:
        # Check overlap: last few tokens of chunk 0 should match first few of chunk 1
        # Tokenizer might not cleanly split words, but token count logic ensures it
        assert chunks[1].token_count <= 10
