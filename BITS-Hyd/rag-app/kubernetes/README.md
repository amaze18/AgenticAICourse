# RAG Application — Kubernetes Deployment

This folder contains the Kubernetes manifests to deploy a self-hosted Retrieval-Augmented
Generation (RAG) stack: **vLLM** (LLM inference), **Qdrant** (vector database), and a
**RAG API** service that orchestrates the two, all running inside a dedicated
`rag-system` namespace, optionally exposed externally via an **Ingress**.

## Architecture at a glance

```
Client
  │
  ▼
Ingress (optional, external access)
  │
  ▼
RAG API  ──────►  Qdrant   (retrieve relevant chunks)
   │
   └───────────►  vLLM     (generate the answer)
```

The RAG API is the orchestrator: on each request it embeds the query, retrieves the
top-matching chunks from Qdrant, then sends the query + retrieved context to vLLM to
generate the final response.

## Files, in apply order

The numeric prefixes indicate the order the manifests are meant to be applied in —
foundational resources first, workloads next, networking last.

### `00-namespace.yaml`
Creates the `rag-system` namespace. Every other resource in this folder lives inside it,
keeping the RAG stack isolated from other workloads in the cluster.

### `01-secrets.yaml`
Template for an optional `hf-secret` holding a Hugging Face access token
(`HF_TOKEN`). Only required if you swap in a **gated** model or want to avoid Hugging
Face's anonymous rate limits — the default model (`Qwen/Qwen3-8B-Instruct`) is public.
This file is a reference/GitOps template; the recommended way to actually create the
secret is via the imperative command shown inside the file (`kubectl create secret
generic ...`) so the token never ends up committed to git in plaintext.

### `02-storage.yaml`
Defines two `PersistentVolumeClaim`s:
- **`hf-cache-pvc`** (50Gi) — caches the downloaded model weights so vLLM doesn't
  re-download the ~16GB model every time its pod restarts.
- **`qdrant-storage-pvc`** (20Gi) — persists Qdrant's vector + payload data across pod
  restarts.

You'll likely need to set `storageClassName` to match a storage class available on
your cluster.

### `10-vllm-deployment.yaml`
Deploys **vLLM**, serving `Qwen/Qwen3-8B-Instruct` through an OpenAI-compatible API on
port 8000. Key details:
- Requires a GPU node — scheduled via `nodeSelector: gpu: rtx4090`, so a node must be
  labeled accordingly, and the NVIDIA device plugin / GPU Operator must be installed.
- Mounts the `hf-cache-pvc` so model weights persist between restarts.
- Uses a memory-backed `emptyDir` for `/dev/shm` (needed for tensor operations).
- `readiness`/`liveness` probes hit `/health` with generous startup delays, since
  loading an 8B model takes time.

### `11-vllm-service.yaml`
A `ClusterIP` Service exposing the vLLM deployment internally as `vllm-service:8000`,
so other pods (the RAG API) can reach it by a stable DNS name.

### `20-qdrant-deployment.yaml`
Deploys **Qdrant**, the vector database that stores document embeddings for
retrieval. Mounts the `qdrant-storage-pvc`, exposes REST (6333) and gRPC (6334) ports,
and has a readiness probe on `/readyz`.

### `21-qdrant-service.yaml`
A `ClusterIP` Service exposing Qdrant's REST and gRPC ports internally as
`qdrant-service`.

### `30-rag-api-deployment.yaml`
Deploys the **RAG API** application itself (2 replicas for availability). This is the
orchestration layer — it doesn't need a GPU. It's configured entirely via environment
variables:

| Env var | Value | Purpose |
|---|---|---|
| `QDRANT_HOST` / `QDRANT_PORT` | `qdrant-service` / `6333` | Where to find Qdrant |
| `QDRANT_COLLECTION` | `documents` | Which Qdrant collection to query |
| `VLLM_BASE_URL` | `http://vllm-service:8000/v1` | Where to find the LLM |
| `VLLM_MODEL_NAME` | `Qwen/Qwen3-8B-Instruct` | Model name to request from vLLM |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for queries |
| `TOP_K` | `4` | Number of chunks retrieved per query |

Before applying this file, you need to build and push your own `rag-api` image and
update the `image:` field (the comment at the top of the file shows the
`docker build` / `docker push` commands).

### `31-rag-api-service.yaml`
A `ClusterIP` Service exposing the RAG API internally on port 80 → container port 8000.

### `40-ingress.yaml`
Optional `Ingress` resource that exposes the RAG API to the outside world (e.g. at
`rag.example.com`) via an nginx ingress controller. Only apply this if you have an
ingress controller installed and want external HTTP access — otherwise you can reach
the RAG API from within the cluster via `rag-api-service`.

## How it all fits together

1. **Namespace, secret, and storage** (`00`–`02`) set up the environment the workloads
   will run in.
2. **vLLM** (`10`–`11`) and **Qdrant** (`20`–`21`) are the two backing services —
   independent, stateful (via PVCs), each with their own Deployment + Service pair.
3. **RAG API** (`30`–`31`) is the stateless orchestrator that ties the two together:
   it embeds the incoming query, retrieves context from Qdrant, and sends a
   context-augmented prompt to vLLM to generate the final answer.
4. **Ingress** (`40`) is the optional front door that lets traffic from outside the
   cluster reach the RAG API.

## Suggested deployment steps

```bash
kubectl apply -f 00-namespace.yaml
kubectl create secret generic hf-secret \
  --namespace rag-system \
  --from-literal=HF_TOKEN=<your-hf-token>   # optional, skip for public models
kubectl apply -f 02-storage.yaml
kubectl apply -f 10-vllm-deployment.yaml
kubectl apply -f 11-vllm-service.yaml
kubectl apply -f 20-qdrant-deployment.yaml
kubectl apply -f 21-qdrant-service.yaml
kubectl apply -f 30-rag-api-deployment.yaml   # after building/pushing your rag-api image
kubectl apply -f 31-rag-api-service.yaml
kubectl apply -f 40-ingress.yaml              # optional, requires an ingress controller
```

## Prerequisites checklist

- [ ] A Kubernetes cluster with at least one GPU node (RTX 4090 or equivalent),
      labeled `gpu=rtx4090`
- [ ] NVIDIA device plugin / GPU Operator installed
- [ ] A storage class available for the PVCs (or set `storageClassName` explicitly)
- [ ] A `rag-api` container image built and pushed to a registry your cluster can pull from
- [ ] (Optional) An nginx ingress controller installed, if you plan to apply `40-ingress.yaml`
