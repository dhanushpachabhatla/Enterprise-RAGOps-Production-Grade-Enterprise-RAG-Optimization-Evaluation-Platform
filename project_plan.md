# Enterprise RAGOps: Production-Grade Enterprise RAG Optimization & Evaluation Platform

## Project Goal

The goal of this project is **not** to build another simple "chat with PDF" application.

The goal is to build a production-grade Enterprise RAG (Retrieval Augmented Generation) platform that replicates how real AI engineering teams design, evaluate, optimize, and deploy LLM-powered knowledge systems.

Modern enterprises contain millions of internal documents distributed across: Slack conversations, emails, engineering documentation, support tickets, CRM records, meeting transcripts, code repositories, wikis, and business documents.

A naive RAG pipeline often fails because of: poor document chunking, weak retrieval, missing context, hallucinations, outdated or conflicting information, poor evaluation practices, high latency, and high inference cost.

This project focuses on systematically improving RAG using experiments and proving every improvement with measurable evaluation metrics.

---

## Core Philosophy

We do **not** add complexity because a technique is popular. Every improvement must answer:

> Did this technique actually improve retrieval accuracy, reduce hallucination, or improve production performance?

**Bad reasoning:** "Added GraphRAG because GraphRAG is trending."

**Good reasoning:** "GraphRAG improved multi-hop relationship queries by 35%, but increased latency by 3x. Therefore, a router agent only sends relationship-based queries to GraphRAG."

The entire project follows an iterative ML experimentation style:

```
Baseline → Measure → Improve → Evaluate → Compare → Deploy
```

---

## Dataset

### Primary Dataset

We will use **EnterpriseRAG-Bench** by Onyx.

Repository: https://github.com/onyx-dot-app/EnterpriseRAG-Bench

EnterpriseRAG-Bench simulates a realistic company, "Redwood Inference," which provides AI model inference services.

Dataset size:
- 500,000+ enterprise documents
- 500 benchmark questions

Documents come from:

| Source | Approx. Documents |
|---|---|
| Slack | 275k |
| Gmail | 120k |
| Linear | 35k |
| Google Drive | 25k |
| Hubspot | 15k |
| Meeting Transcripts | 10k |
| GitHub | 8k |
| Jira | 6k |
| Confluence | 5k |

This provides realistic enterprise challenges: duplicate information, internal terminology, outdated documents, conflicting knowledge, cross-document reasoning, and noisy conversations.

---

## Question Categories Available

EnterpriseRAG-Bench provides the following question categories.

**Basic Questions** — simple single-document retrieval. Tests basic chunking, embedding quality, retrieval accuracy.

**Semantic Questions** — questions without exact keyword overlap. Tests semantic search, embeddings, query rewriting.

**Intra-document Reasoning** — information exists far apart inside the same document. Tests hierarchical retrieval, context engineering.

**Project Related Questions** — requires combining multiple project documents. Tests multi-hop retrieval, GraphRAG.

**Constrained Questions** — many relevant documents exist but only one satisfies conditions. Tests reranking, metadata filtering.

**Conflicting Information** — documents contradict each other. Tests freshness handling, reasoning, answer verification.

**Completeness Questions** — need multiple documents for a complete answer. Tests recall optimization.

**High-Level Questions** — no single document contains the answer. Tests agentic reasoning, knowledge synthesis.

**Info Not Found** — answer does not exist. Tests hallucination prevention, guardrails.

---

## Overall System Architecture

```
Enterprise Documents
        │
        ▼
Document Ingestion Pipeline
        │
        ▼
Chunking Experiment Layer
  (Fixed / Semantic / Hierarchical / Semantic+Hierarchical)
        │
        ▼
Embedding Generation
        │
        ▼
Vector Database (Qdrant)
        │
        ▼
Hybrid Retrieval Engine
  (Dense Retrieval + BM25)
        │
        ▼
Reranking
        │
        ▼
Context Engineering
        │
        ▼
LLM Generation
        │
        ▼
Answer Verification
        │
        ▼
Guardrails
        │
        ▼
Evaluation Engine
        │
        ▼
Monitoring Dashboard
```

---

## Phase 1: Data Ingestion Pipeline

**Goal:** Convert all enterprise data into a universal format.

**Input:** Slack JSON, emails, GitHub discussions, Jira tickets, documents.

**Output:**

```json
{
    "doc_id": "123",
    "source": "slack",
    "created_at": "date",
    "author": "employee",
    "content": "...",
    "metadata": {}
}
```

**Implement:** document parsing, metadata extraction, deduplication, cleaning, normalization.

---

## Phase 2: Baseline RAG System

Build the simplest possible RAG.

**Architecture:**

```
Document → Fixed Token Chunking → Embedding Model → Vector DB
  → Top-K Retrieval → LLM → Answer
```

**Purpose:** create baseline numbers.

**Metrics — Retrieval:** Recall@1, Recall@5, Recall@10, Precision@K, MRR, NDCG

**Metrics — Generation:** Faithfulness, answer correctness, hallucination rate

**Metrics — System:** Latency, token usage, cost/query

---

## Phase 3: Chunking Experiments

**Goal:** find the optimal chunking strategy.

**Experiment A — Fixed Chunking**
Examples: 256 tokens, 512 tokens, 1024 tokens. Measure performance.

**Experiment B — Semantic Chunking**
Split based on meaning. Test whether preserving semantic boundaries improves retrieval.

**Experiment C — Hierarchical Chunking**

```
Document → Section → Paragraph
```

Retrieve smaller chunks; send larger parent context. Measure accuracy improvement and token increase.

**Experiment D — Semantic + Hierarchical**
Combine both. Expected to yield a production-level chunking strategy.

---

## Phase 4: Hybrid Retrieval

**Problem:** dense vectors fail on exact matching (ticket IDs, error codes, names, dates).

**Solution:** combine vector search + BM25 search + score fusion.

**Evaluate:** dense vs. hybrid retrieval.

---

## Phase 5: Reranking

**Pipeline:**

```
Retrieve Top 50 Documents → Cross Encoder Reranker → Return Top 5
```

**Measure:** MRR improvement, context precision improvement, additional latency.

---

## Phase 6: Context Engineering

**Goal:** optimize information sent to the LLM.

**Context Compression** — remove unnecessary content; measure token reduction and accuracy change.

**Context Ordering** — test highest-relevance-first, chronological order, and reordered context; evaluate the lost-in-the-middle problem.

**Dynamic Context Selection** — instead of always retrieving top 10, use 3 chunks for easy queries and 15 chunks for complex queries.

**Query Rewriting** — convert vague queries (e.g. "deployment issue yesterday") into specific ones (e.g. "production deployment failure incident logs date"); measure retrieval improvement.

**Multi-Query Retrieval** — generate multiple query variations, retrieve, merge; measure recall increase.

---

## Phase 7: Metadata-Aware RAG

Enterprise queries require metadata. Example: "What did Sarah say about deployment last month?" needs an author filter, time filter, and source filter.

**Implement:** a Query Metadata Extraction Agent, e.g.:

```json
{
    "author": "Sarah",
    "time": "last month",
    "source": "slack"
}
```

Use metadata filtering before retrieval.

---

## Phase 8: GraphRAG

GraphRAG should only solve relationship problems.

**Extract entities:** employees, projects, teams, services, documents.

**Extract relations**, for example:
- Employee `owns` Service
- Service `depends_on` Database
- Project `created_by` Team

**Store:** Neo4j.

**Evaluate:**
- Normal questions → dense RAG should win.
- Relationship questions → GraphRAG should win.

**Goal:** prove when GraphRAG is actually useful.

---

## Phase 9: Agentic RAG System

Use agents only after retrieval optimization.

**Architecture:**

```
User Question
      │
      ▼
Router Agent → Classify Query Type
      │
      ├── Simple Query        → Dense RAG
      ├── Exact Lookup        → Hybrid Search
      └── Relationship Query  → GraphRAG
      │
      ▼
Answer Generation
      │
      ▼
Verification Agent
      │
      ▼
Final Answer
```

---

## Phase 10: Self-Correcting RAG

Implement a reflection loop.

```
Retrieve → Generate Answer → Judge Answer
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                                ▼
            Is grounded? YES               Is grounded? NO
                  │                                │
               Return                  Rewrite Query → Retrieve Again
```

**Measure:** hallucination reduction.

---

## Phase 11: Guardrails

Production systems require safety.

**Input Guardrails** — detect prompt injection, malicious queries, sensitive information.

**Retrieval Guardrails** — check whether enough evidence was found; if confidence is low, do not answer.

**Output Guardrails** — validate that every claim is supported by retrieved context, preventing hallucinations.

---

## Phase 12: Evaluation Framework

Use automated evaluation.

**Tools:** RAGAS, LLM-as-Judge, custom metrics.

**Track — Retrieval:** Recall@K, Precision, MRR, NDCG

**Track — Generation:** Faithfulness, Relevance, Correctness

**Track — Production:** latency, throughput, cost

---

## Phase 13: Experiment Tracking

Maintain a leaderboard so every architecture decision has numbers behind it:

| Pipeline | Recall@10 | Faithfulness | Hallucination | Latency |
|---|---|---|---|---|
| Basic RAG | | | | |
| Semantic Chunking | | | | |
| Hybrid Search | | | | |
| Reranking | | | | |
| GraphRAG | | | | |

---

## Phase 14: Model Strategy

**Embeddings (run locally):** BGE, E5, Nomic Embed

**Generation (local):** Qwen 7B, Llama 8B

**API models** — used only for evaluation, comparison, and as judge.

**Compare:**

| Model | Accuracy | Latency | Cost |
|---|---|---|---|
| Local 7B | | | |
| API Model | | | |

**Goal:** optimize the quality/cost tradeoff.

---

## Phase 15: Production Engineering

- **Backend:** FastAPI
- **Vector Database:** Qdrant
- **Graph Database:** Neo4j
- **Database:** PostgreSQL
- **Queue:** Redis + Celery
- **Containerization:** Docker
- **Deployment:** Cloud / Kubernetes

---

## Phase 16: Monitoring

**Track production metrics:** requests/minute, latency, failed retrievals, hallucination rate, user feedback, token usage, cost.

**Tools:** Prometheus, Grafana, LangSmith.

---

## Final Goal

By the end we should prove measurable improvement, for example:

| Stage | Recall@10 | Hallucination |
|---|---|---|
| Baseline | 70% | 15% |
| Optimized | 92% | 4% |

And understand *why*:
- Semantic chunking improved semantic queries.
- Hybrid retrieval improved exact lookup.
- GraphRAG improved relationship queries.
- Context engineering reduced cost.
- Guardrails reduced hallucinations.

The final product is not just a chatbot — it is a complete Enterprise RAG experimentation, optimization, and production platform.
