# Production LLM Serving Engines: Boilerplate & Configurations

In a real enterprise environment (like NatWest on AWS), you **do not write Python API servers from scratch** like we did locally. 

Instead, you use highly optimized, pre-compiled C++/CUDA Docker containers provided by vLLM or Hugging Face. Your job as a Platform Engineer is to write the **Infrastructure as Code (Kubernetes YAML or Docker commands)** to configure and deploy these containers onto GPU machines.

Here is the exact boilerplate code used in the industry for each engine.

---

## 1. vLLM (The King of Open-Source Serving)
vLLM is deployed as a Docker container that automatically exposes an OpenAI-compatible API.

**How to implement in Kubernetes (Deployment YAML):**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm-container
        image: vllm/vllm-openai:latest
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args: [
          "--model", "meta-llama/Meta-Llama-3-8B",
          "--tensor-parallel-size", "2",    # Split model across 2 GPUs
          "--gpu-memory-utilization", "0.90", # Use 90% of GPU VRAM for PagedAttention
          "--max-model-len", "4096"         # Max context window
        ]
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 2  # Request exactly 2 Nvidia GPUs from the K8s cluster
```

---

## 2. Hugging Face TGI (Text Generation Inference)
TGI is the enterprise standard from Hugging Face. It is heavily used for models requiring built-in quantization or Multi-LoRA support.

**How to implement via Docker Run (Standard AWS EC2 deployment):**
```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Meta-Llama-3-8B \
  --num-shard 4 \           # Tensor Parallelism (Split across 4 GPUs)
  --quantize bitsandbytes \ # Apply 8-bit quantization on the fly
  --max-concurrent-requests 100
```
*Interview Note:* The `--num-shard` flag is critical. If a model doesn't fit on one GPU, you shard it across multiple GPUs.

---

## 3. DJL DeepSpeed (AWS SageMaker)
When deploying massive models (e.g., 70B+ parameters) on AWS SageMaker, you use the Deep Learning Java Library (DJL) combined with Microsoft DeepSpeed. You don't write K8s YAML for this; instead, you provide a `serving.properties` configuration file to SageMaker.

**How to implement (serving.properties file):**
```properties
# This file is zipped and uploaded to AWS S3, then passed to the SageMaker Endpoint
engine=DeepSpeed
option.model_id=meta-llama/Meta-Llama-3-70B
option.tensor_parallel_degree=8     # Distribute across 8 GPUs instantly
option.max_input_len=2048
option.max_output_len=1024
option.dtype=fp16                   # Use 16-bit precision to save memory
```

---

## 4. KServe (Kubernetes Serverless ML)
KServe is a Kubernetes operator. Instead of manually writing a K8s Deployment + Service + Ingress (which takes 100 lines of YAML), KServe abstracts it into a single `InferenceService` file and handles auto-scaling automatically (even scaling GPUs down to 0 when nobody is using the API).

**How to implement (InferenceService CRD YAML):**
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-vllm-predictor
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 5  # Automatically scale up to 5 GPU pods during high traffic
    containers:
      - name: kserve-container
        image: vllm/vllm-openai:latest
        args:
          - --model=meta-llama/Meta-Llama-3-8B
          - --port=8080
        resources:
          limits:
            nvidia.com/gpu: 1
```

