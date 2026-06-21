import spacy
import tiktoken
import hashlib
from typing import List, Dict, Any
from src.chunking.fixed import Chunk

class SemanticChunker:
    def __init__(self, model_name: str = "gpt-3.5-turbo", chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        # We use a blank English pipeline instead of the heavy en_core_web_sm model.
        # This completely bypasses the Neural Networks (tok2vec, parser, NER) which were bottlenecking the CPU.
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    def chunk_batch(self, documents: List[Any]) -> List[Chunk]:
        """Processes an entire batch of documents simultaneously using Spacy's optimized C-pipe."""
        valid_docs = [d for d in documents if d.content.strip()]
        if not valid_docs:
            return []
            
        texts = [d.content for d in valid_docs]
        
        # The performance optimization: process entire batch via C-level pipes
        parsed_docs = list(self.nlp.pipe(texts, batch_size=256))
        
        all_chunks = []
        for doc_obj, parsed_doc in zip(valid_docs, parsed_docs):
            sentences = [sent.text.strip() for sent in parsed_doc.sents if sent.text.strip()]
            if not sentences:
                continue
                
            # HUGE CPU OPTIMIZATION: Encode all sentences at once using Tiktoken's Rust-compiled multi-threading
            encoded_sents = self.tokenizer.encode_batch(sentences, disallowed_special=())
            sent_token_counts = [len(enc) for enc in encoded_sents]
            
            chunks = []
            current_chunk_sents = []
            current_tokens = 0
            
            for sentence, sent_tokens in zip(sentences, sent_token_counts):
                if current_tokens + sent_tokens > self.chunk_size and current_chunk_sents:
                    chunk_text = " ".join(current_chunk_sents)
                    chunks.append(self._create_chunk(doc_obj.doc_id, chunk_text, doc_obj.metadata))
                    
                    # Dynamic Semantic Overlap: Backtrack until we hit self.overlap tokens
                    overlap_sents = []
                    overlap_tokens = 0
                    # We iterate backwards through the chunk we just flushed
                    for s in reversed(current_chunk_sents):
                        s_toks = len(self.tokenizer.encode(s, disallowed_special=()))
                        if overlap_tokens + s_toks > self.overlap and overlap_sents:
                            break
                        overlap_sents.insert(0, s)
                        overlap_tokens += s_toks
                        
                    current_chunk_sents = overlap_sents + [sentence]
                    # Encode exactly once at the boundary
                    current_tokens = len(self.tokenizer.encode(" ".join(current_chunk_sents), disallowed_special=()))
                else:
                    current_chunk_sents.append(sentence)
                    current_tokens += sent_tokens
                    
            if current_chunk_sents:
                chunk_text = " ".join(current_chunk_sents)
                chunks.append(self._create_chunk(doc_obj.doc_id, chunk_text, doc_obj.metadata))
                
            all_chunks.extend(chunks)
            
        return all_chunks

    def chunk_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Legacy single-document processing. Slower."""
        if not text.strip():
            return []
            
        metadata = metadata or {}
        # Parse grammatical sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks = []
        current_chunk_sents = []
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = len(self.tokenizer.encode(sentence, disallowed_special=()))
            
            if current_tokens + sent_tokens > self.chunk_size and current_chunk_sents:
                # Flush the chunk
                chunk_text = " ".join(current_chunk_sents)
                chunks.append(self._create_chunk(doc_id, chunk_text, metadata))
                
                # Semantic overlap: Keep the final sentence of the previous chunk for context continuity
                current_chunk_sents = [current_chunk_sents[-1], sentence]
                current_tokens = len(self.tokenizer.encode(" ".join(current_chunk_sents), disallowed_special=()))
            else:
                current_chunk_sents.append(sentence)
                current_tokens += sent_tokens
                
        # Flush any remaining sentences
        if current_chunk_sents:
            chunk_text = " ".join(current_chunk_sents)
            chunks.append(self._create_chunk(doc_id, chunk_text, metadata))
            
        return chunks

    def _create_chunk(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> Chunk:
        tokens = self.tokenizer.encode(text, disallowed_special=())
        chunk_id = hashlib.md5(text.encode("utf-8")).hexdigest()
        return Chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=text,
            metadata=metadata,
            token_count=len(tokens)
        )
