# LangChain + LangGraph: 5-Day Learning Plan (Production-Focused)

**Context:** Intermediate Python, new to LLMs. Building a **large-scale, zero-hallucination RAG pipeline (10M+ docs)** for Target: hybrid retrieval (BM25 + embeddings) → ANN retrieval + reranking → source confidence scoring → constrained generation → citation-backed responses, with continuous evals, caching, and observability as first-class concerns. Provider infra TBD (possibly Azure OpenAI / Bedrock / direct API). Goal: understand LangChain/LangGraph AND how to productionalize this specific kind of pipeline — reliability, observability, deployment, scaling.

## Recommended Frameworks / Industry-Standard Stack (as of mid-2026)

Any new agent picking up this file should assume the following as the default stack unless the Notes/Decisions section says otherwise:

| Layer | Standard choice(s) | Notes |
|---|---|---|
| Orchestration (linear) | **LangChain (LCEL)** | Use for retrieval chains, prompt composition — anything without branching |
| Orchestration (stateful/agentic) | **LangGraph** | Use for the confidence-threshold branch (answer vs. "insufficient evidence"), checkpointing, human-in-the-loop |
| Retrieval | Hybrid search: BM25 + dense embeddings, `EnsembleRetriever` (LangChain) | Two-stage retrieval (hybrid → rerank) is the accepted 2026 default and outperforms hybrid-alone by ~17% recall in published benchmarks |
| Reranker | **Voyage AI rerank-2.5** or **Cohere Rerank v3.5** (managed); **BGE reranker v2** (open-source/self-hosted) | Reranking is considered a near-mandatory step at this scale, not optional |
| Vector DB | **Pinecone** or **Azure AI Search** (managed); **Weaviate** or **Qdrant** (open-source) | Must support hybrid search + metadata filtering at 10M+ doc scale |
| Evaluation | **RAGAS** (faithfulness, relevance, context precision) + **LangSmith Evaluations** (regression testing) | RAGAS gives RAG-specific metrics; LangSmith ties evals to your actual traces |
| Observability | **LangSmith** (native, tightly integrated) or **Langfuse** (leading open-source alternative) | Needed to trace retrieval path, chunk scores, and confidence-gate decisions per your diagram |
| Persistence/checkpointing | LangGraph's Postgres or Redis checkpointer | Required for resumable, auditable agent runs |

This table reflects the current industry-standard pattern as of when this plan was written — re-verify before relying on it if this file is reused much later, since this ecosystem moves fast.

**Format:** 1–2 focused hours/day. Each day = concept block + hands-on exercise + production note.

> How to use this doc: paste this whole file at the start of any new Claude chat when working on this plan, then say which day/section you're on. Update the "Notes / Decisions" section at the bottom as you learn specifics about the actual Target project. This file itself should live in the GitHub repo (see below) so it stays versioned alongside the code it produces.

## Working With Your Repo (Learn-in-Public Convention)

You've started a GitHub repo with just a README. Treat it as the single source of truth that matures alongside this plan — don't treat the 5 days as throwaway tutorial code and the "real" repo as a separate future effort. Every day's hands-on work gets committed.

**Suggested structure**, organized by pipeline stage rather than by day (so it can grow directly into the production layout instead of needing a rewrite later):

```
agentic-ai-learning/
├── README.md                 # updated each day: what exists, what doesn't yet
├── docs/
│   ├── learning-plan.md      # this file, kept in sync as it evolves
│   └── decisions.md          # mirrors the "Notes / Decisions" section below
├── notebooks/                # exploratory work, one per day
│   ├── day1_foundations.ipynb
│   ├── day2_retrieval.ipynb
│   ├── day3_langgraph.ipynb
│   ├── day4_reliability.ipynb
│   └── day5_deployment.ipynb
├── src/
│   ├── ingestion/            # Day 2 — normalize/dedupe/chunk
│   ├── retrieval/            # Day 2 — hybrid retriever, reranker
│   ├── graph/                # Day 3 — LangGraph state/nodes/edges
│   ├── reliability/          # Day 4 — confidence scoring, constrained generation, citations
│   └── api/                  # Day 5 — FastAPI or LangGraph Platform entrypoint
├── tests/                    # Day 4 onward — structural/behavioral + adversarial tests
├── configs/                  # model/provider config, kept out of code
└── requirements.txt          # pinned versions from Day 1 onward
```

**Conventions to start now, not later:**
- One commit (or small PR) per day's hands-on exercise — small, reviewable diffs, not one giant end-of-week dump
- `README.md` gets a short update each day: what's implemented, what's still a stub, what's TBD
- Move exploratory notebook code into `src/` once it's stable enough to be reused — the notebook is where you learn, `src/` is what the system actually runs
- `requirements.txt` pinned from Day 1 (per your existing production note) so the repo is reproducible from the very first commit
- Treat `docs/decisions.md` as the running log for the "Notes / Decisions" section below — copy updates there as you go, so the repo (not just this file) captures why choices were made

---

## Day 1 — LLM Foundations + LangChain Core

**Concepts**
- LLM basics: tokens, context window, temperature/top_p, system vs user vs assistant roles
- Why a framework at all (vs raw API calls): standardization, composability, swappable components
- Core LangChain abstractions: `ChatModel`, `PromptTemplate`, `OutputParser`, `Runnable`, LCEL (`|` pipe syntax)
- Structured output with Pydantic (critical for production — never trust raw text parsing)

**Hands-on**
- Install `langchain`, `langchain-openai` (or provider of choice), set up `.env` for API keys
- Build a simple LCEL chain: prompt → model → structured Pydantic output parser
- Swap the underlying model provider without touching the chain logic (this is the point of the abstraction)
- **Repo checkpoint:** commit `requirements.txt` (pinned versions), your first chain in `src/`, and a one-paragraph README update describing what exists so far

**Production note**
- Never hardcode API keys — use environment variables / secrets manager from day one
- Pin your `langchain` package versions — this ecosystem moves fast and breaks often

---

## Day 2 — Retrieval, Tools & Memory (with an eye on Hybrid Retrieval + Reranking)

**Concepts**
- Embeddings and vector stores (semantic search) vs BM25 (keyword/lexical search) — and why production RAG at scale usually uses **both** (hybrid retrieval), since embeddings miss exact terms (SKUs, IDs, names) that BM25 catches
- Basic RAG pipeline: load → split → embed → store → retrieve → generate
- **Reranking**: a fast ANN step gets top-K candidates cheaply; a slower, more accurate cross-encoder reranker (e.g., MiniLM, BGE reranker) reorders those candidates by true relevance — this two-stage "cheap recall, expensive precision" pattern is standard at 10M+ doc scale
- Tool/function calling: how the model decides to call external functions
- Memory patterns: conversation buffer vs summarization vs external state (why in-memory chat history doesn't scale)

**Hands-on**
- Build a minimal RAG pipeline over a small local doc set (use an in-memory vector store like FAISS or Chroma for now)
- Add a BM25 retriever alongside your vector retriever and combine results (LangChain's `EnsembleRetriever` does exactly this)
- Try a cross-encoder reranker on the combined results and observe how the ordering changes
- **Repo checkpoint:** commit the hybrid retriever + reranker under `src/ingestion/` and `src/retrieval/`; update README

**Production note**
- Chunking strategy, metadata, and retrieval quality matter more than model choice — bad chunks = bad answers regardless of model
- At 10M+ docs, ingestion needs deduplication, format normalization, metadata extraction, and versioning *before* indexing — garbage in, garbage retrieved
- Local vector stores (FAISS/Chroma) are fine for learning; production needs a managed vector DB (pgvector, Pinecone, Azure AI Search, etc.) with hybrid search support and access control

---

## Day 3 — LangGraph: Agents & Orchestration

**Concepts**
- Why LangGraph exists: LangChain chains are DAGs (one direction); LangGraph adds **state**, **cycles**, and **conditional branching** — needed for anything agentic
- Core concepts: `State`, `Node`, `Edge`, conditional edges, checkpointing
- Single-agent loop (ReAct-style: reason → act → observe → repeat)
- Multi-agent patterns (supervisor/worker, handoffs) — at a conceptual level

**Hands-on**
- Build a simple LangGraph graph: 2–3 nodes, one conditional edge, a loop that terminates on a condition
- Add a tool-calling node and trace the state as it flows through the graph
- **Repo checkpoint:** commit the graph under `src/graph/`; note the state schema in `docs/decisions.md`

**Production note**
- Checkpointing (LangGraph's persistence layer) is what lets you resume, replay, and debug agent runs — treat this as non-negotiable for anything beyond a demo
- Design the state schema deliberately (Pydantic/TypedDict) — it's your contract between nodes and your main debugging surface

---

## Day 4 — Reliability, Grounding & Testing (Zero-Hallucination Focus)

**Concepts**
- Failure modes specific to LLM systems: hallucination, malformed output, infinite tool loops, silent drift after model/version updates
- **Source confidence scoring**: scoring retrieved chunks by freshness, trust, overlap, and retrieval consistency, then filtering out low-confidence chunks *before* generation — this is what keeps weak evidence from ever reaching the model
- **Constrained generation**: prompting (and structurally enforcing) that the model may only answer from retrieved context, never from parametric/external knowledge — plus a confidence threshold that triggers an explicit "insufficient evidence found" response instead of a guess
- **Citation-backed responses**: requiring every claim in the output to be traceable to a specific chunk/doc/timestamp, so answers are auditable after the fact
- Guardrails: input validation, output validation (Pydantic schema enforcement), retry-with-correction loops
- Evaluation for this architecture specifically: adversarial queries, retrieval recall benchmarks, hallucination-rate tracking, offline + online evals (LangSmith Evaluations or custom pytest-based evals)
- Fallback strategies: model fallback chains, timeout handling, circuit breakers

**Hands-on**
- Add a confidence-scoring step to your Day 2 retrieval output: filter/rank chunks by a simple heuristic (recency + retrieval score) before passing them to the LLM
- In your prompt, explicitly instruct the model to answer only from provided context and to say "insufficient evidence" if confidence is below a threshold — test it with a query you know isn't well covered by your docs
- Require the model's output to include citations (chunk IDs) tied to a Pydantic schema, and validate that every claim maps to a real citation
- Write 3–5 adversarial test cases: including at least one query designed to have no good answer in your doc set, and confirm the system correctly refuses rather than hallucinating
- **Repo checkpoint:** commit confidence scoring + constrained generation under `src/reliability/`, adversarial cases under `tests/`

**Production note**
- You cannot unit-test LLM output for exact strings — test for structure, constraints, and behavior instead
- Hallucination-rate tracking should be a standing metric, not a one-time check — re-run your adversarial eval set every time you change a prompt, model, or retrieval config
- Version your prompts like code (git, changelogs) — a "small prompt tweak" is a production change and should go through the same review process as code

---

## Day 5 — Observability, Caching, Deployment & Scaling

**Concepts**
- **Observability everywhere**: trace the full retrieval path (which chunks were retrieved, their rankings/scores, token attribution per step), not just the final answer — this is what makes a "why did it say this" investigation possible, and it's what "auditable" citations depend on operationally
- **Caching + memory layer**: caching frequent queries and retrieval results (not just final answers) saves cost/latency at 10M+ doc scale; separately, user/session memory supports personalization without re-retrieving from scratch every turn
- Deployment patterns: wrap chain/graph in FastAPI, containerize (Docker), or use LangGraph Platform/LangServe if applicable to Target's stack
- Scaling: async execution, request batching, rate limiting against provider quotas
- Security: secrets management, PII handling/redaction, prompt injection defenses, output sanitization before it hits downstream systems
- CI/CD: what changes trigger re-testing (prompt change, model version bump, tool schema change, reranker/embedding model swap)

**Hands-on**
- Wrap your Day 3 graph in a minimal FastAPI endpoint
- Add structured logging/tracing that captures, per request: retrieved chunk IDs + scores, confidence score, whether the "insufficient evidence" path fired, tokens used, and latency
- Add a simple cache (even an in-memory dict keyed by normalized query) in front of retrieval and observe the latency difference on repeated queries
- **Repo checkpoint:** commit the API entrypoint under `src/api/`; this is the point where `README.md` should describe the whole system end-to-end, not just a stub

**Capstone checklist (review before calling anything "production-ready")**
- [ ] Secrets are not hardcoded
- [ ] Output is schema-validated, not trusted raw
- [ ] Retries/fallbacks exist for model and tool failures
- [ ] Prompts are versioned and reviewed like code
- [ ] Tracing/logging exists for cost, latency, and errors
- [ ] There's a regression test suite that runs before deploy
- [ ] Rate limits and timeouts are handled explicitly
- [ ] There's a plan for what happens when the underlying model is upgraded/deprecated

---

## Notes / Decisions (fill in as you learn more about the Target project)

- **Actual use case:** Large-scale RAG over 10M+ docs, zero-hallucination target, citation-backed answers with an explicit "insufficient evidence" fallback
- **Pipeline stages (from architecture diagram):** ingest/normalize → hybrid retrieval (BM25 + embeddings) → ANN retrieval + reranking → source confidence scoring → constrained generation → citation-backed response, with continuous evals, caching/memory, and observability as cross-cutting concerns
- **LLM provider/infra:** _TBD (Azure OpenAI / Bedrock / direct API)_
- **Vector DB / data sources:** _TBD (needs to support hybrid search at 10M+ doc scale — e.g., Azure AI Search, pgvector, Pinecone)_
- **Reranker model:** _TBD (diagram mentions MiniLM / BGE reranker as examples)_
- **Deployment target:** _TBD (internal API, Kubernetes, serverless, etc.)_
- **Observability tooling available at Target:** _TBD_

## Resources
- LangChain docs: https://python.langchain.com/
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- LangSmith (tracing/evals): https://docs.smith.langchain.com/
