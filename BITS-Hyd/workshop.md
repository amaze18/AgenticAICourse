# Production-Grade RAG Chat Systems: Architecture & Deployment
### A Hands-On Course by AIGuruKul Foundation
*Empowering minds to learn, develop and grow together.*

---

## Course Description

This course teaches engineers how to design, size, and deploy a production-grade Retrieval-Augmented Generation (RAG) chat system serving 40–50 concurrent users on a single GPU. Using a real reference architecture — Qwen3-30B-A3B served via vLLM, Qdrant for vector storage, and a FastAPI/Gunicorn/Nginx serving stack — students learn to reason quantitatively about GPU memory, concurrency limits, and failure modes, rather than treating LLM infrastructure as a black box.

**Format:** 10 modules · ~18 hours of instruction · labs after every module
**Delivery:** Lecture + guided lab + capstone project

---

## Who This Course Is For

- Backend/ML engineers moving an LLM prototype into production for the first time
- Platform engineers responsible for GPU capacity planning
- Technical leads who need to size infrastructure and set SLOs for an AI product

## Prerequisites

- Comfortable with Python and reading Docker Compose files
- Basic familiarity with REST APIs and Linux command line
- No prior GPU or LLM-serving experience required — memory math is taught from first principles

## Learning Outcomes

By the end of this course, students will be able to:
1. Calculate GPU VRAM budgets for model weights, KV cache, and execution overhead
2. Explain how weight quantization (AWQ) preserves accuracy while shrinking memory footprint
3. Configure and reason about vLLM's PagedAttention scheduler and concurrency limits
4. Design a retrieval pipeline with a vector database, including chunking and metadata strategy
5. Instrument a production LLM service with Prometheus metrics and alerting rules
6. Explain the difference between CPU-bound and GPU-bound concurrency, and size worker processes accordingly
7. Assemble the full stack — Nginx, Gunicorn/Uvicorn, FastAPI, vLLM, Qdrant — into a coherent, defensible architecture

---

## Module 1 — System Architecture Overview
**Duration: 1 hour**

- The four-layer mental model: edge, application, inference, storage
- Reading a target workload spec: concurrent users, context length, hardware
- Why memory discipline and deterministic concurrency are the two design principles that recur throughout the course

**Lab:** Annotate a blank architecture diagram with the four layers using a provided workload spec.

---

## Module 2 — Model Quantization Fundamentals
**Duration: 2 hours**

- Why 4-bit quantization is necessary for serving 30B-parameter models on a single GPU
- The accuracy-vs-memory tradeoff, and where naive quantization fails (salient channel clipping)
- AWQ (Activation-aware Weight Quantization): the coordinate scaling and transformation approach
- AWQ vs. GPTQ: calibration cost vs. accuracy recovery strategy

**Lab:** Given a weight matrix and a calibration set, manually compute a per-channel scaling factor and verify it preserves activation magnitude.

---

## Module 3 — GPU Memory Budgeting
**Duration: 2.5 hours**

- Building a full VRAM budget line by line: embedding model, model weights, execution overhead, KV cache
- The KV cache byte-per-token formula and why it's usually the *largest* allocation on the card
- Calculating maximum token capacity from a fixed memory pool
- Host RAM vs. GPU VRAM: what belongs where, and why they scale independently

**Lab:** Given a GPU spec and a model configuration, build a complete memory budget table and identify the safety margin.

---

## Module 4 — Concurrency & the PagedAttention Scheduler
**Duration: 2 hours**

- Compute-bound vs. memory-bound: why chunked prefill changes the schedule, not the memory reservation
- Deriving a safe concurrent-user ceiling from token capacity and per-session footprint
- What happens at KV cache saturation, and how the scheduler responds
- KV cache precision optimization (FP8) and its tradeoffs on long conversations
- Mitigation playbook: alerting thresholds, per-request token budgets

**Lab:** Given a target concurrency number, work backward to the required KV cache allocation and validate it against the memory budget from Module 3.

---

## Module 5 — Retrieval Pipeline & Vector Storage
**Duration: 2.5 hours**

- Qdrant as a CPU-resident storage engine: why it lives outside the GPU memory budget
- Semantic chunking strategy: chunk size, overlap, and why boundaries matter
- Metadata enrichment and contextual inversion: preserving provenance for the embedding model
- HNSW index configuration: `m`, `ef_construct`, `full_scan_threshold` and their tradeoffs

**Lab:** Chunk a sample policy document two ways (naive fixed-size vs. recursive with overlap) and compare retrieval quality on boundary-spanning queries.

---

## Module 6 — Observability & Production Metrics
**Duration: 2 hours**

- The four observability zones: GPU, inference engine, application, vector store
- Key metrics: `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_factor`, TTFT, `DCGM_FI_DEV_FB_USED`, Qdrant search latency
- Writing alerting rules with meaningful thresholds, not arbitrary ones
- Escalation design: single-signal warnings vs. multi-signal critical pages

**Lab:** Write a Prometheus scrape config and a set of alerting rules for the reference architecture; simulate a cache-saturation scenario and verify the alert fires.

---

## Module 7 — End-to-End Data Flow & Deployment Topology
**Duration: 1.5 hours**

- Tracing a single request: gateway → embed → retrieve → generate → stream
- Container topology: which services share a network, which touch the GPU
- Docker Compose as an infrastructure contract between services

**Lab:** Diagram the container topology for a given docker-compose.yml and mark which services are GPU-bound vs. CPU-bound.

---

## Module 8 — The FastAPI Orchestration Layer
**Duration: 2 hours**

- Responsibilities of the orchestration layer: context injection, prompt templating, routing
- Async route handlers and why they matter for streaming responses
- Token generation as an async generator: enabling low time-to-first-token

**Lab:** Implement an async FastAPI route that streams tokens from a mock generator without buffering the full response.

---

## Module 9 — Concurrency Model: Gunicorn, Workers & the GIL
**Duration: 2 hours**

- Gunicorn's master/worker model, and the restaurant-manager analogy
- Why the GIL makes threading ineffective for CPU-bound work, and how multiple worker processes route around it
- Synchronous vs. asynchronous workers: which one fits an I/O-bound RAG endpoint
- Sizing workers to CPU cores, not to concurrent users — and why the GPU, not the CPU, is the true bottleneck

**Lab:** Load-test a sync-worker deployment vs. an async-worker deployment at the same concurrency level and compare throughput.

---

## Module 10 — Perimeter Architecture & Capstone
**Duration: 2.5 hours**

- Nginx's role: SSL termination, connection buffering, rate limiting, static asset offloading
- Defense in depth: how Nginx, Gunicorn, and vLLM's scheduler each guard against a different failure mode
- **Capstone project:** Given a new workload spec (different GPU, different concurrency target), students produce a complete design package:
  - GPU memory budget table
  - Concurrency ceiling calculation with safety margin
  - Alerting rule set
  - Container topology diagram
  - Written justification for every sizing decision

**Deliverable:** A one-page design document defended in a 10-minute review session, graded against the reference architecture's design principles.

---

## Assessment Structure

| Component | Weight |
|---|---|
| Module labs (10 × short exercises) | 40% |
| Mid-course quiz (Modules 1–5) | 15% |
| Capstone design package | 35% |
| Capstone defense/review | 10% |

## Suggested Reference Materials

- vLLM documentation (PagedAttention, scheduler configuration)
- Qdrant documentation (HNSW parameters, collection configuration)
- AWQ paper (activation-aware weight quantization)
- FastAPI + Uvicorn async documentation
- Prometheus alerting best practices guide
