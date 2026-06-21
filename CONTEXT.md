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

| Metric | Baseline (Fixed 512) | Exp 1A (Semantic 512) | Exp 1B (Hierarchical 128) |
|--------|----------------------|----------------------------|----------------------------|
| **Recall@1** | 0.428 | 0.428 | 0.396 |
| **Recall@5** | 0.556 | 0.548 | 0.538 |
| **Recall@10** | 0.610 | **0.610** | 0.584 |
| **Precision@5**| 0.1904 | **0.2144** | 0.1364 |
| **NDCG@10** | 0.6097 | **0.6829** | 0.4521 |
| **MRR** | 0.4831 | 0.4798 | 0.4548 |

### Ablation Study Conclusion (Step 1)
- **Winner**: Semantic Chunking (Exp 1A). It maximizes Ranking Quality (NDCG) without destroying the context window.
- **Loser**: Hierarchical Chunking (Exp 1B). 128 tokens causes severe Context Fragmentation, destroying the Dense Vector's ability to understand the paragraph's overall meaning.

## Recent Changes
- Completed Phase 3 Evaluation framework.
- Wired LLM generation specifically to local LM Studio via `http://127.0.0.1:1234/v1`.
