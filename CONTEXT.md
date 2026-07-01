# Enterprise RAGOps — Project Context

## Project Summary
Building a production-grade Enterprise RAG (Retrieval Augmented Generation) platform that replicates how real AI engineering teams design, evaluate, optimize, and deploy LLM-powered knowledge systems. Focused on systematic improvement through measurable metrics.

## Architecture Overview
The system currently consists of three distinct pipelines:
1. **Data Ingestion**: Parses enterprise data into a universal schema (`src/ingestion`).
2. **Baseline RAG**: Uses Fixed Token Chunking (512 tokens), `sentence-transformers` (`BAAI/bge-small-en-v1.5`) for 384-dimensional dense vectors, `qdrant-client` for Vector storage, and LM Studio (`mistral-7b`) for generation (`src/chunking`, `src/embeddings`, `src/retrieval`, `src/generation`, `src/pipeline`).
3. **Evaluation Engine**: Computes mathematically rigorous retrieval metrics like Recall@K and MRR over the 500-question benchmark dataset (`src/evaluation`).

## Current Phase
Moving into **Phase 4: Advanced RAG System** (Implementing Hybrid Retrieval, Semantic Chunking, and Re-ranking to improve baseline metrics) or fulfilling missing metric engines.

## Codebase Map
src/
├── chunking/     (fixed.py)
├── embeddings/   (embedder.py)
├── evaluation/   (dataset.py, metrics.py)
├── generation/   (llm.py connecting to LM Studio)
├── ingestion/    (models.py, parser.py)
├── pipeline/     (baseline.py, evaluate.py)
└── retrieval/    (vector_db.py)

## Key Design Decisions
- Excluded `all_documents/` from agent parsing to save 50M+ tokens.
- Opted for `.jsonl` extraction to lazily load and parse the 2.49GB dataset without RAM issues.
- Integrated local `qdrant` rather than a managed service for faster local prototyping.
- Integrated `LM Studio` locally for generation to avoid API costs.

## Data Notes
- **512K docs**, all `.txt`, in `all_documents/`
- **JSONL Corpus**: Extracted successfully to `enterprise_corpus.jsonl` (2.49 GB).
- **Benchmark:** `all_documents/questions.jsonl` (500 questions).

## Current Metrics
Tested against 50,000 document subset using `BAAI/bge-small-en-v1.5`.

| Metric | Baseline | Exp 1A (Semantic) | Exp 1C (Sliding) | Exp 2A (Hybrid 50/50) | Exp 3 (Re-ranking) | Exp 4 (HyDE Dense) |
|--------|----------|-------------------|------------------|-----------------------|--------------------|--------------------|
| **Recall@1** | 0.428 | 0.428 | 0.436 | 0.456 | **0.552** | 0.394 |
| **Recall@5** | 0.556 | 0.548 | 0.562 | 0.614 | **0.692** | 0.526 |
| **Recall@10** | 0.610 | 0.610 | 0.620 | 0.672 | **0.726** | 0.588 |
| **Precision@5**| 0.1904 | 0.2144 | 0.1464 | 0.1592 | **0.1780** | 0.2112 |
| **NDCG@10** | 0.6097 | **0.6829** | 0.4864 | 0.5000 | 0.6095 | 0.6438 |
| **MRR** | 0.4831 | 0.4798 | 0.4902 | 0.4965 | **0.6104** | 0.4516 |

### Phase 1 Ablation Study Conclusion
- **Overall Winner**: **Semantic Chunking (Exp 1A)**. The massive drop in Ranking Quality (NDCG 0.48) caused by Sliding Window's noise is not worth the tiny 0.01 bump in Recall. Semantic Chunking's clean grammatical boundaries provide the most mathematically pure Dense Vectors.

### Phase 4 Hybrid Retrieval Conclusion
- **Success**: Recall@10 leaped from the absolute mathematical ceiling of 0.610 up to **0.672**! BM25 definitively solved the "exact keyword matching" problem (e.g., finding exact ticket IDs or error codes).
- **The New Flaw**: BM25 relies purely on word-counting. It retrieves completely irrelevant documents just because they happen to share common words. When we fused the scores equally using RRF, BM25's "word-matching noise" polluted the top results, dragging our pristine 0.68 NDCG (Ranking Quality) down to 0.50.
- **RRF Weighting Fails**: We tested weighting Dense at 80% and Sparse at 20% (Exp 2B). We lost a massive amount of Recall (dropped from 0.672 back to 0.646), but our NDCG barely recovered (0.50 to 0.51). You cannot statically weight away BM25 noise without destroying its benefits.

### Phase 5 Cross-Encoder Re-ranking Conclusion
- **Total Success**: We used Hybrid (50/50) to fetch the Top 30 unique documents to maximize Recall, then passed them to `BAAI/bge-reranker-base`.
- **Recall Exploitation**: By providing 30 chunks instead of 10, the Re-ranker found the true answer hidden in the bottom ranks and pulled it up, pushing Recall@10 from 0.672 to an astonishing **0.726** (+11.6% over baseline).
- **NDCG Recovery**: The Cross-Encoder successfully identified the BM25 noise and banished it, restoring NDCG from 0.500 back to **0.6095**.
- **MRR Victory**: The correct answer is now appearing at Rank #1 exactly **55.2%** of the time (up from 42.8%), pushing MRR to an all-time high of **0.6104**!

### Phase 6 Context Engineering (HyDE) Conclusion
- **Hybrid Failure**: We first tested HyDE against the Hybrid (BM25 + Dense) pipeline. Recall@10 collapsed from 0.726 down to **0.448**. Feeding a 250-word hallucinated paragraph into a BM25 sparse index flooded it with common filler words ("the", "system"), completely destroying exact keyword matching.
- **Dense-Only Failure**: Even when strictly isolated to Pure Dense Vectors (avoiding BM25 interference), HyDE still failed to beat the Semantic Baseline. Recall@10 dropped from 0.610 to **0.588**.
- **Why it failed**: In Enterprise RAG (Slack, Jira, GitHub), questions are highly specific factual lookups (e.g. "Why did build 489 fail?"). Because the LLM does not actually know the answer, it hallucinates a fake but plausible reason (e.g. "It failed due to a memory leak"). The Vector Database embeds this fake reason, and retrieves documents about "memory leaks", completely missing the true document that might state "build 489 failed due to NPM registry outage". HyDE is dangerous for exact factual Enterprise lookups.

## Recent Changes
- Completed Phase 3 Evaluation framework.
- Wired LLM generation specifically to local LM Studio via `http://127.0.0.1:1234/v1`.
