import spacy
import tiktoken
import hashlib
from typing import List, Any
from src.chunking.fixed import Chunk

class HierarchicalChunker:
    """
    Experiment 1B: Hierarchical (Parent-Child) Chunking.
    Highly optimized: Runs the Spacy sentencizer exactly ONCE per document, 
    and simultaneously slides a Child-Window and a Parent-Window over the parsed sentences!
    """
    def __init__(self, parent_size: int = 512, child_size: int = 128):
        self.parent_size = parent_size
        self.child_size = child_size
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        
    def _create_chunk(self, doc_id: str, text: str, metadata: dict) -> Chunk:
        chunk_id = hashlib.md5(text.encode("utf-8")).hexdigest()
        return Chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=text,
            metadata=metadata,
            token_count=0
        )
        
    def chunk_batch(self, documents: List[Any]) -> List[Chunk]:
        valid_docs = [d for d in documents if d.content.strip()]
        if not valid_docs: 
            return []
            
        texts = [d.content for d in valid_docs]
        
        # We only run spacy.pipe ONCE on the original documents!
        parsed_docs = list(self.nlp.pipe(texts, batch_size=256))
        
        all_child_chunks = []
        for doc_obj, parsed_doc in zip(valid_docs, parsed_docs):
            sentences = [sent.text.strip() for sent in parsed_doc.sents if sent.text.strip()]
            if not sentences: 
                continue
                
            # Tiktoken Rust batch encoding
            encoded_sents = self.tokenizer.encode_batch(sentences, disallowed_special=())
            sent_token_counts = [len(enc) for enc in encoded_sents]
            
            parent_sents = []
            parent_tokens = 0
            
            child_sents = []
            child_tokens = 0
            children_of_current_parent = []
            
            def flush_child(sents):
                children_of_current_parent.append(" ".join(sents))
                
            def flush_parent(p_sents):
                parent_text = " ".join(p_sents)
                parent_chunk_id = hashlib.md5(parent_text.encode("utf-8")).hexdigest()
                
                # Apply parent metadata to all children that were formed during this parent's window
                for child_text in children_of_current_parent:
                    new_meta = doc_obj.metadata.copy() if doc_obj.metadata else {}
                    new_meta["parent_chunk_id"] = parent_chunk_id
                    new_meta["parent_text"] = parent_text
                    all_child_chunks.append(self._create_chunk(doc_obj.doc_id, child_text, new_meta))
                
                children_of_current_parent.clear()

            for sentence, sent_tokens in zip(sentences, sent_token_counts):
                # 1. Check if adding this sentence overflows the 512-token Parent Window
                if parent_tokens + sent_tokens > self.parent_size and parent_sents:
                    if child_sents:
                        flush_child(child_sents)
                        child_sents = []
                        child_tokens = 0
                    
                    flush_parent(parent_sents)
                    
                    # Parent Semantic Overlap
                    parent_sents = [parent_sents[-1], sentence]
                    parent_tokens = len(self.tokenizer.encode(" ".join(parent_sents), disallowed_special=()))
                    
                    # The new child naturally starts with the parent's overlap sentence
                    child_sents = [parent_sents[-1], sentence]
                    child_tokens = parent_tokens
                else:
                    parent_sents.append(sentence)
                    parent_tokens += sent_tokens
                    
                    # 2. Check if adding this sentence overflows the 128-token Child Window
                    if child_tokens + sent_tokens > self.child_size and child_sents:
                        flush_child(child_sents)
                        # Child Semantic Overlap
                        child_sents = [child_sents[-1], sentence] 
                        child_tokens = len(self.tokenizer.encode(" ".join(child_sents), disallowed_special=()))
                    else:
                        child_sents.append(sentence)
                        child_tokens += sent_tokens
                        
            # Flush remaining stragglers
            if child_sents:
                flush_child(child_sents)
            if parent_sents:
                flush_parent(parent_sents)
                
        return all_child_chunks
