# RAG on Kubernetes with Qwen3-8B-Instruct (vLLM) + Qdrant

This guide walks through building a complete Retrieval-Augmented Generation (RAG) stack on a single-node Kubernetes cluster. It starts from a bare Ubuntu workstation with an RTX 4090 GPU and ends with a working RAG API that uses Qwen3-8B-Instruct (served through vLLM) for generation and Qdrant for vector search.

## Architecture Overview

The system is organized as a small set of services inside a Kubernetes namespace. An optional ingress sits in front of a FastAPI-based RAG API, which embeds incoming queries on CPU, retrieves relevant context from Qdrant, and then calls vLLM (running on the GPU node) to generate the final answer.

```
                 +----------------------+
                 |      Ingress         |  (optional)
                 +----------+-----------+
                            |
                    rag-api-service
                            |
                     RAG API (FastAPI)
                     - embeds queries (sentence-transformers, CPU)
                     - retrieves context from Qdrant
                     - calls vLLM for generation
          +-----------------+-----------------+
          |                                   |
      vllm-service                     qdrant-service
   Qwen3-8B-Instruct (vLLM)               Qdrant
          |                                   |
     RTX 4090 GPU node                Persistent Volume
```

---

## Part 0: Host Prerequisites

### 0.1 Confirm the GPU and Drivers Work

Before touching Docker or Kubernetes, confirm the GPU itself is usable. Run `nvidia-smi` and check that it lists your RTX 4090 along with a driver version — if this step fails, nothing downstream will work until the NVIDIA drivers are fixed.

```bash
nvidia-smi
```

### 0.2 Install Docker Engine

Docker provides the container runtime that both minikube and your application images will rely on. This step adds Docker's official APT repository and keyring, installs the Docker Engine and CLI packages, and grants your user permission to run Docker without `sudo`.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER
newgrp docker
```

If `newgrp docker` doesn't take effect in your current shell, log out and back in, or reboot. Once done, verify the install with a test container.

```bash
docker run hello-world
```

### 0.3 Install the NVIDIA Container Toolkit

This toolkit is what allows Docker containers — and later, Kubernetes pods — to actually access the GPU hardware. Without it, containers would only see a CPU, regardless of what's physically installed.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

After restarting Docker, confirm it can see the GPU by running `nvidia-smi` from inside a container — you should see the same output as step 0.1, just sourced from within Docker.

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 0.4 Install minikube

minikube provisions a lightweight, single-node Kubernetes cluster locally, which is what the rest of this guide deploys onto. The command below installs it only if it isn't already present.

```bash
which minikube || (
  curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
  sudo install minikube-linux-amd64 /usr/local/bin/minikube
)
minikube version
```

### 0.5 Install kubectl

`kubectl` is the command-line tool used to interact with the Kubernetes cluster for every step that follows. As with minikube, this only installs it if it's missing.

```bash
which kubectl || (
  curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
)
kubectl version --client
```

---

## Part 1: Start the Cluster

### 1.1 Start minikube with GPU Support

This single command spins up a Kubernetes cluster inside Docker and automatically wires up the NVIDIA device plugin, so the cluster is GPU-aware from the start.

```bash
minikube start --driver docker --container-runtime docker --gpus all
```

### 1.2 Verify the GPU Is Visible to Kubernetes

It's worth double-checking that Kubernetes itself recognizes the GPU as a schedulable resource before deploying anything. You're looking for `"nvidia.com/gpu": "1"` under the node's allocatable resources.

```bash
kubectl get nodes
kubectl get nodes -o json | grep -A5 "nvidia.com/gpu"
```

### 1.3 Label the Node

Labeling the node lets later deployment manifests target it specifically via a `nodeSelector`, ensuring GPU-hungry workloads like vLLM land on the right machine.

```bash
kubectl label node minikube gpu=rtx4090
```

---

## Part 2: Deploy the Stack

All commands in this section assume you're inside the `rag/` project directory, which contains the `kubernetes/` manifests and the `rag-api/` source folder.

### 2.1 Namespace and Storage

This creates an isolated namespace for all RAG-related resources and provisions the persistent volumes that vLLM and Qdrant will use to store model weights and vector data.

```bash
kubectl apply -f kubernetes/00-namespace.yaml
kubectl apply -f kubernetes/02-storage.yaml
kubectl -n rag-system get pvc
```

Wait until both PVCs report `Bound`. If they remain `Pending`, check the available storage classes and update `storageClassName` in `kubernetes/02-storage.yaml` accordingly — though minikube's default `standard` storage class usually binds without any changes needed.

```bash
kubectl get storageclass
```

### 2.2 Optional: Hugging Face Token Secret

This step is only necessary if the Qwen3-8B-Instruct model becomes gated or you start hitting anonymous rate limits when downloading it. The secret stores your token so vLLM can authenticate with Hugging Face.

```bash
kubectl create secret generic hf-secret \
  --namespace rag-system \
  --from-literal=HF_TOKEN=<your-hf-token>
```

If you use this, remember to uncomment the corresponding `HF_TOKEN` environment block in `kubernetes/10-vllm-deployment.yaml`.

### 2.3 Deploy vLLM (Qwen3-8B-Instruct)

This deploys the vLLM inference server running Qwen3-8B-Instruct along with its associated Kubernetes service. Since the first startup downloads roughly 16 GB of model weights, expect this step to take several minutes.

```bash
kubectl apply -f kubernetes/10-vllm-deployment.yaml
kubectl apply -f kubernetes/11-vllm-service.yaml
kubectl -n rag-system get pods -w
```

Watch the pod's progress via its logs, and press Ctrl+C once it shows `Running` and `1/1` ready. Subsequent restarts reuse the weights cached on the PVC, so they start much faster.

```bash
kubectl -n rag-system logs deployment/vllm -f
```

### 2.4 Deploy Qdrant

This brings up the Qdrant vector database and its service, which the RAG API will query for relevant document chunks at request time.

```bash
kubectl apply -f kubernetes/20-qdrant-deployment.yaml
kubectl apply -f kubernetes/21-qdrant-service.yaml
kubectl -n rag-system get pods
```

### 2.5 Build the RAG API Image Inside minikube's Docker Environment

minikube runs its own internal Docker daemon, separate from the host's. Pointing your shell at that daemon via `minikube docker-env` lets you build the image directly where the cluster can find it, avoiding the need for a remote registry.

```bash
eval $(minikube docker-env)
cd rag-api
docker build -t myrepo/rag-api:v1 .
cd ..
```

Afterward, make sure the deployment manifest references this locally built image and tells Kubernetes not to attempt a registry pull.

```yaml
      - name: rag-api
        image: myrepo/rag-api:v1
        imagePullPolicy: IfNotPresent
```

### 2.6 Deploy the RAG API

With the image built and the manifest updated, deploy the RAG API and its service. This is the FastAPI layer that ties embedding, retrieval, and generation together.

```bash
kubectl apply -f kubernetes/30-rag-api-deployment.yaml
kubectl apply -f kubernetes/31-rag-api-service.yaml
kubectl -n rag-system get pods
```

### 2.7 Confirm Everything Is Up

A final sanity check across all resources in the namespace. You want to see `vllm`, `qdrant`, and both `rag-api` pods reporting `Running` and ready.

```bash
kubectl -n rag-system get all
```

---

## Part 3: Ingest Documents

Before you can query anything, documents need to be embedded and loaded into Qdrant. This forwards the Qdrant service to your local machine, installs the required Python packages, and runs the ingestion script against a folder of documents.

```bash
kubectl -n rag-system port-forward svc/qdrant-service 6333:6333 &
pip install qdrant-client sentence-transformers
python rag-api/ingest.py --docs-dir ./my-docs --recreate
```

Point `--docs-dir` at a folder of `.txt`/`.md` files you want to query over. Drop the `--recreate` flag on later runs if you're only adding new documents to an existing collection rather than rebuilding it from scratch.

---

## Part 4: Query the RAG API

With ingestion complete, forward the RAG API service locally and send a question to it as JSON. The API embeds the query, retrieves matching context from Qdrant, and returns a generated answer along with its supporting sources.

```bash
kubectl -n rag-system port-forward svc/rag-api-service 8080:80 &

curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "your question here"}'
```

The response includes both the generated answer and the source chunks that informed it, each with a relevance score.

```json
{
  "answer": "...",
  "sources": [
    {"text": "...", "score": 0.83, "payload": {"text": "...", "source": "file.md"}}
  ]
}
```

Two additional endpoints are available for basic health checking.

```bash
curl http://localhost:8080/health
curl http://localhost:8080/ready
```

---

## Tuning Notes for a Single RTX 4090 (24 GB)

The `--gpu-memory-utilization 0.90` and `--max-model-len 16384` flags in `kubernetes/10-vllm-deployment.yaml` are safe starting points for an 8B model in bfloat16 on 24 GB of VRAM. Lower `max-model-len` if you see out-of-memory errors in the vLLM logs, or raise it if you have headroom and need longer contexts.

The embedding model (`all-MiniLM-L6-v2`) runs entirely on CPU inside the RAG API pods, so it never competes with vLLM for GPU memory. Because of this, `rag-api` runs 2 replicas since it's cheap to scale on CPU, while `vllm` stays at a single replica since only one pod can use the GPU at a time.

Since minikube shares resources with whatever the host allocates to it, pods that get evicted or OOM-killed are often a sign minikube itself needs more memory or CPU. Check its current allocation, or restart it with higher limits.

```bash
minikube config view
# or restart with more resources, e.g.:
minikube start --driver docker --container-runtime docker --gpus all --memory 16g --cpus 6
```

---

## Troubleshooting

| Symptom | Likely Cause / Fix |
|---|---|
| `connection to server localhost:8080 refused` on any `kubectl` command | No cluster running, or kubeconfig not pointing at one — run `minikube start` and check `kubectl config current-context` |
| `PROVIDER_DOCKER_NOT_FOUND` from minikube | Docker isn't installed — see Part 0.2 |
| `nvidia.com/gpu` missing from `kubectl get nodes -o json` | GPU device plugin isn't active — re-run `minikube start --gpus all` and confirm Part 0.3 succeeded |
| vllm pod stuck `Pending` | Usually no node satisfies `nodeSelector: gpu: rtx4090`, or the GPU resource isn't schedulable — check `kubectl -n rag-system describe pod <vllm-pod>` |
| vllm pod `CrashLoopBackOff` with OOM in logs | Lower `--max-model-len` or `--gpu-memory-utilization` in `kubernetes/10-vllm-deployment.yaml` |
| PVCs stuck `Pending` | Set `storageClassName` in `kubernetes/02-storage.yaml` to a class from `kubectl get storageclass` |
| rag-api pod `ImagePullBackOff` | You built the image without `eval $(minikube docker-env)` first, or forgot `imagePullPolicy: IfNotPresent` |
| `/query` returns empty sources | Nothing has been ingested yet, or you're querying the wrong `QDRANT_COLLECTION` — check Part 3 |
