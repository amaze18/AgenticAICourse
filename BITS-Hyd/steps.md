RAG on Kubernetes with Qwen3-8B-Instruct (vLLM) + Qdrant

Complete setup guide: from a bare Ubuntu workstation with an RTX 4090, to a running RAG API backed by Qwen3-8B-Instruct served via vLLM and Qdrant for vector search.

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

Part 0: Host prerequisites
0.1 Confirm the GPU and drivers work

nvidia-smi

You should see your RTX 4090 listed with a driver version. If this fails, fix your NVIDIA drivers before continuing (nothing below will work without this).
0.2 Install Docker Engine

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

Let your user run Docker without sudo:

sudo usermod -aG docker $USER
newgrp docker

(If newgrp docker doesn't take effect in your current shell, log out and back in, or reboot.)

Verify:

docker run hello-world

0.3 Install the NVIDIA Container Toolkit

This lets Docker (and later, Kubernetes pods) access the GPU.

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

Verify Docker can see the GPU:

docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

You should see the same GPU output as step 0.1, but now from inside a container.
0.4 Install minikube

which minikube || (
  curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
  sudo install minikube-linux-amd64 /usr/local/bin/minikube
)
minikube version

0.5 Install kubectl (if not already installed)

which kubectl || (
  curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
)
kubectl version --client

Part 1: Start the cluster
1.1 Start minikube with GPU support

minikube start --driver docker --container-runtime docker --gpus all

This provisions a single-node Kubernetes cluster inside Docker and wires up the NVIDIA device plugin automatically.
1.2 Verify the GPU is visible to Kubernetes

kubectl get nodes
kubectl get nodes -o json | grep -A5 "nvidia.com/gpu"

You should see "nvidia.com/gpu": "1" under the node's allocatable resources.
1.3 Label the node

kubectl label node minikube gpu=rtx4090

Part 2: Deploy the stack

All commands below assume you're in the rag/ project directory (the one containing kubernetes/, rag-api/, and this README).
2.1 Namespace and storage

kubectl apply -f kubernetes/00-namespace.yaml
kubectl apply -f kubernetes/02-storage.yaml
kubectl -n rag-system get pvc

Wait until both PVCs show Bound. If they stay Pending, check available storage classes and set storageClassName in kubernetes/02-storage.yaml accordingly:

kubectl get storageclass

(minikube ships a default standard storage class, so this usually binds without any changes.)
2.2 Optional: Hugging Face token secret

Only needed if Qwen/Qwen3-8B-Instruct becomes gated or you hit anonymous rate limits:

kubectl create secret generic hf-secret \
  --namespace rag-system \
  --from-literal=HF_TOKEN=<your-hf-token>

Then uncomment the HF_TOKEN env block in kubernetes/10-vllm-deployment.yaml.
2.3 Deploy vLLM (Qwen3-8B-Instruct)

kubectl apply -f kubernetes/10-vllm-deployment.yaml
kubectl apply -f kubernetes/11-vllm-service.yaml
kubectl -n rag-system get pods -w

Ctrl+C once the vllm pod shows Running and 1/1 ready. The first startup downloads ~16 GB of model weights, so this can take several minutes — watch progress with:

kubectl -n rag-system logs deployment/vllm -f

Subsequent pod restarts reuse the cached weights on the PVC and start much faster.
2.4 Deploy Qdrant

kubectl apply -f kubernetes/20-qdrant-deployment.yaml
kubectl apply -f kubernetes/21-qdrant-service.yaml
kubectl -n rag-system get pods

2.5 Build the RAG API image inside minikube's Docker environment

minikube has its own internal Docker daemon, so point your build at that instead of pushing to a remote registry:

eval $(minikube docker-env)
cd rag-api
docker build -t myrepo/rag-api:v1 .
cd ..

In kubernetes/30-rag-api-deployment.yaml, make sure the container spec includes:

      - name: rag-api
        image: myrepo/rag-api:v1
        imagePullPolicy: IfNotPresent

(IfNotPresent tells Kubernetes to use the locally-built image instead of trying to pull from a registry.)
2.6 Deploy the RAG API

kubectl apply -f kubernetes/30-rag-api-deployment.yaml
kubectl apply -f kubernetes/31-rag-api-service.yaml
kubectl -n rag-system get pods

2.7 Confirm everything is up

kubectl -n rag-system get all

You want vllm, qdrant, and both rag-api pods showing Running and ready.
Part 3: Ingest documents

kubectl -n rag-system port-forward svc/qdrant-service 6333:6333 &
pip install qdrant-client sentence-transformers
python rag-api/ingest.py --docs-dir ./my-docs --recreate

Point --docs-dir at a folder of .txt/.md files you want to query over. Drop the --recreate flag on subsequent runs if you're just adding more documents to an existing collection.
Part 4: Query the RAG API

kubectl -n rag-system port-forward svc/rag-api-service 8080:80 &

curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "your question here"}'

Response shape:

{
  "answer": "...",
  "sources": [
    {"text": "...", "score": 0.83, "payload": {"text": "...", "source": "file.md"}}
  ]
}

Optional health checks:

curl http://localhost:8080/health
curl http://localhost:8080/ready

Tuning notes for a single RTX 4090 (24 GB)

    --gpu-memory-utilization 0.90 and --max-model-len 16384 in kubernetes/10-vllm-deployment.yaml are safe starting points for an 8B model in bfloat16 on 24 GB. Lower max-model-len if you see OOM errors in kubectl -n rag-system logs deployment/vllm; raise it if you have headroom and need longer contexts.

    The embedding model (all-MiniLM-L6-v2) runs on CPU inside the rag-api pods — it doesn't touch the GPU, which stays reserved for vLLM.

    rag-api runs 2 replicas since it's cheap (CPU-only); vllm stays at 1 replica since only one pod can use the single GPU.

    With minikube, resource limits in the manifests are shared with whatever you gave minikube itself. If pods get evicted or OOM-killed, check minikube's allocated resources:

    minikube config view
    # or restart with more resources, e.g.:
    minikube start --driver docker --container-runtime docker --gpus all --memory 16g --cpus 6

Troubleshooting
Symptom 	Likely cause / fix
connection to server localhost:8080 refused on any kubectl command 	No cluster running, or kubeconfig not pointing at one — run minikube start and kubectl config current-context
PROVIDER_DOCKER_NOT_FOUND from minikube 	Docker isn't installed — see Part 0.2
nvidia.com/gpu missing from kubectl get nodes -o json 	GPU device plugin isn't active — re-run minikube start --gpus all and confirm Part 0.3 succeeded
vllm pod stuck Pending 	Usually no node satisfies nodeSelector: gpu: rtx4090 or GPU resource isn't schedulable — check kubectl -n rag-system describe pod <vllm-pod>
vllm pod CrashLoopBackOff with OOM in logs 	Lower --max-model-len or --gpu-memory-utilization in kubernetes/10-vllm-deployment.yaml
PVCs stuck Pending 	Set storageClassName in kubernetes/02-storage.yaml to a class from kubectl get storageclass
rag-api pod ImagePullBackOff 	You built the image without eval $(minikube docker-env) first, or forgot imagePullPolicy: IfNotPresent
/query returns empty sources 	Nothing has been ingested yet, or you're querying the wrong QDRANT_COLLECTION — check Part 3
