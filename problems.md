# Enterprise RAGOps - Engineering Problem Log

This document tracks all critical technical bottlenecks, edge cases, and algorithmic problems encountered while building the production RAG pipeline, and the exact engineering solutions used to solve them.

---

### 1. Tiktoken Special Token Crashing
- **Problem**: When parsing raw enterprise text, `tiktoken` occasionally encounters literal strings like `<|endoftext|>`. Because this matches an internal OpenAI control token, `tiktoken.encode()` throws a fatal `ValueError`.
- **Solution**: Explicitly set the parameter `disallowed_special=()` in `tokenizer.encode()`. This forces the tokenizer to treat all text as raw strings, preventing fatal crashes during ingestion.

### 2. Spacy CPU Bottleneck (Single-Threaded Python Loops)
- **Problem**: During Semantic Chunking, running `self.nlp(text)` inside a Python `for` loop caused ingestion speed to drop from 18 it/s down to 1.8 it/s. Python loops are single-threaded, and the CPU could not parse the grammar fast enough to keep the GPU fed with vectors.
- **Solution**: Refactored the architecture to use `list(self.nlp.pipe(texts, batch_size=256))`. This utilizes Spacy's internal C-compiled multi-threading to process massive batches of documents simultaneously across all CPU cores.

### 3. Spacy Neural Network Overhead
- **Problem**: Even after using `nlp.pipe()`, chunking was still crawling. This was because Spacy defaults to loading its heavy 30MB neural networks (Dependency Parser, Tagger, NER) which are useless for simple sentence boundary detection.
- **Solution**: Initialized the model using `spacy.blank("en")` and explicitly added only the `"sentencizer"` pipe. This bypassed the neural networks entirely and relied purely on C-optimized regex and punctuation rules, instantly multiplying speed by 10x.

### 4. Tiktoken Loop Overhead
- **Problem**: To ensure chunks stayed under 512 tokens, the code was running `tiktoken.encode(sentence)` on every single sentence iteratively. Across 50,000 documents containing ~100 sentences each, this meant calling the tokenizer 5,000,000+ times from Python, causing massive overhead.
- **Solution**: Utilized Tiktoken's highly optimized multi-threaded Rust backend by calling `tokenizer.encode_batch(sentences)` before the loop began. This encoded thousands of sentences simultaneously in Rust, dropping the overhead to near zero.

### 5. Hierarchical Double-Pass O(2N) Execution
- **Problem**: In Hierarchical Chunking, the naive approach was to run the text through Spacy to build the 512-token Parent chunks, and then pass those chunks *back* into Spacy to build the 128-token Child chunks. This forced the CPU to parse the grammar of every document twice (O(2N)).
- **Solution**: Rewrote the chunker into a custom Single-Pass O(N) algorithm. It parses the document once, and mathematically maintains two sliding windows (a 512-token parent window and a 128-token child window), dynamically flushing boundaries as it iterates through the sentence list.

### 6. Qdrant Local SQLite Disk I/O Bottleneck
- **Problem**: Qdrant's Local Python mode uses SQLite to persist data. When shoving over 500,000 chunks into the database, SQLite's B-Tree index rebalancing caused massive disk swapping. The speed plummeted to 1.01 it/s because the RTX 3060 GPU was sitting idle waiting for the hard drive to finish writing.
- **Solution**: Bypassed SQLite entirely by initializing Qdrant with `QdrantClient(location=":memory:")`. By merging ingestion and evaluation into a single `run_hierarchical_end_to_end.py` script, the entire 500k vector database lived in ~1GB of system RAM, executing at maximum GPU speed and gracefully deleting itself after the evaluation completed.

### 7. Recall Dilution via Granular Chunk Duplication
- **Problem**: At 128 tokens, a highly relevant document is shattered into dozens of tiny chunks. If the Vector DB searches for `top_k=10`, it might return 10 chunks that *all* belong to the exact same document. Because our Recall@10 metric expects to evaluate 10 unique documents, testing against 1 unique document artificially tanks the score.
- **Solution**: In `evaluate_hierarchical.py`, the pipeline intentionally fetches `top_k=50` chunks from Qdrant, and mathematically deduplicates them by `doc_id` until it isolates the absolute Top 10 *Unique* documents. This guarantees a mathematically fair Recall evaluation against the Baseline.

### 8. Python SQLite Vector Search Degradation
- **Problem**: When testing the Native Qdrant Hybrid pipeline, querying 186k Dense points and 182k Sparse points simultaneously brought evaluation speed down to 20 seconds per query (a 3-hour evaluation). Qdrant's `Local Mode` uses basic Python SQLite file reads rather than its optimized Rust memory engine. 
- **Solution**: For rapid local prototyping, we bypassed Qdrant's Sparse index entirely and fell back to computing BM25 natively in Python via `rank_bm25`, bringing the evaluation back down to 36 minutes. In a true production environment, Qdrant must be run inside a Docker container (Rust), which will drop the 20-second search latency to 2 milliseconds.

### 9. HyDE Interference with BM25 Exact Matching
- **Problem**: In Phase 6, we attempted to improve Recall by concatenating the original query with a 250-word LLM-generated Hypothetical Document (HyDE). This caused a catastrophic Recall drop from 72.6% down to 44.8%. Because we fed the massive 250-word paragraph into our Hybrid Pipeline, the BM25 Sparse Index began word-counting all of the hallucinated filler words ("the", "system", "failed", "document"). This completely destroyed BM25's ability to find exact matches for Ticket IDs or specific error codes, flooding the top results with irrelevant documents.
- **Solution**: HyDE is strictly a Semantic (Dense Vector) optimization technique. It must never be passed into a TF-IDF/BM25 sparse index. In an advanced Agentic architecture, a Router Agent should classify the query: if it is a conceptual query, route to HyDE + Dense Vector; if it is a lookup query (IDs, error codes), route to pure BM25.
