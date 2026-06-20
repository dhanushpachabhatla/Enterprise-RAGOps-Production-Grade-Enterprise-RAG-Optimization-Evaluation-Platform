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
- **Recall@K & MRR**: Evaluation engine is built, true metrics pending user running the full dataset insertion via `scripts/embed_corpus.py`.

## Recent Changes
- Completed Phase 3 Evaluation framework.
- Wired LLM generation specifically to local LM Studio via `http://127.0.0.1:1234/v1`.
