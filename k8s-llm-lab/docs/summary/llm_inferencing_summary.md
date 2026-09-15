# LLM Inferencing — Complete Summary
**Quick Reference Guide | NatWest GenAI Platform Engineering**

---

## Section 1: LLM Serving Engines at a Glance

| Engine | Primary Use Case | Key Feature | When to Use | AWS Service | Azure Service |
|---|---|---|---|---|---|
| **vLLM** | High-throughput CPU/GPU serving | PagedAttention (KV-Cache) | Maximum concurrent users, lowest latency | EKS + GPU nodes | AKS + GPU nodes |
| **TGI (Hugging Face)** | Multi-LoRA, enterprise serving | Multi-LoRA + built-in quantization | Multiple fine-tuned models on one GPU | EKS / SageMaker | AKS / Azure Machine Learning (AML) |
| **DJL DeepSpeed** | Massive model deployment | Tensor Parallelism (splits model across GPUs) | Models too big for one GPU (70B+) | SageMaker | AML (Custom Container) / AKS |
| **KServe** | K8s autoscaling orchestrator | Autoscale GPU pods up/down on traffic | Wraps vLLM/TGI, manages scaling | EKS | AKS |

---

## Section 2: Inference Optimization Techniques

| Technique | What It Does | Benefit | Tools |
|---|---|---|---|
| **Quantization (INT8)** | Compresses weights from 16-bit → 8-bit | Halves GPU memory usage | bitsandbytes, TGI flag `--quantize` |
| **Quantization (FP16)** | Uses 16-bit instead of 32-bit | 2x memory reduction, no accuracy loss | PyTorch `torch_dtype=float16` |
| **GGUF (Q4_K_M)** | 4-bit compression for CPU inference | Runs 7B models on a laptop CPU | llama.cpp, Ollama |
| **PagedAttention** | Dynamic KV-Cache memory allocation | 3x more users on the same GPU | vLLM only |
| **Continuous Batching** | Ejects finished requests instantly, adds new ones | No idle GPU time between batches | vLLM, TGI |
| **Tensor Parallelism** | Splits model vertically across N GPUs | Run models larger than one GPU's VRAM | DJL DeepSpeed, vLLM `--tensor-parallel-size` |
| **Pipeline Parallelism** | Splits model layers horizontally across GPUs | Alternative split strategy for huge models | DJL DeepSpeed |

---

## Section 3: Fine-Tuning Techniques

| Technique | What It Does | Benefit | Tools |
|---|---|---|---|
| **Full Fine-Tuning** | Updates ALL model weights | Highest accuracy | PyTorch Trainer |
| **LoRA** | Freezes base model, trains tiny adapter matrices | Trains 0.1% of parameters, same accuracy | Hugging Face PEFT |
| **QLoRA** | LoRA + 4-bit quantization combined | Fine-tunes 70B model on a single GPU | PEFT + bitsandbytes |
| **Multi-LoRA** | Multiple LoRA adapters on one base model | 100 use cases, 1 GPU cluster | TGI `--enable-lora` |

---

## Section 4: Multi-LoRA Architecture (NatWest Hub-and-Spoke)

| Component | Role | Technology |
|---|---|---|
| **Base Model (S3)** | Central model weights, downloaded once | `s3://base-models/llama-3-8b/` |
| **Adapter Store (S3)** | 100+ LoRA adapters, each ~50MB | `s3://lora-adapters/<use-case>/` |
| **Adapter Registry** | Maps `adapter_id` → full S3 path | TGI `--lora-ids` flag OR DynamoDB |
| **TGI Serving Engine** | Loads base model, swaps adapters dynamically | `ghcr.io/huggingface/text-generation-inference` |
| **LRU Cache (VRAM)** | Keeps "hot" adapters in GPU memory | Built into TGI |
| **GenAI Gateway** | Routes requests, tags `adapter_id` | Custom FastAPI / AWS API Gateway |
| **KServe** | Autoscales TGI pods 1 → 10 based on traffic | `InferenceService` CRD on EKS |

**Adapter Resolution Flow:**
```
User Request → GenAI Gateway tags adapter_id 
  → TGI looks up registry → Resolves S3 path 
    → Checks VRAM cache → Download if not cached 
      → Fuse with base model → Return response
```

---

## Section 5: CPU vs GPU — Key Differences

| Aspect | CPU (e.g., Intel i5) | GPU (e.g., Nvidia A100) |
|---|---|---|
| **Cores** | 2 to 16 cores | 6,912 CUDA cores |
| **Strength** | Complex sequential logic | Massively parallel math |
| **Memory Bandwidth** | ~50 GB/s | ~2,000 GB/s |
| **LLM Inference Speed** | 1-3 tokens/sec (TinyLlama) | 50-200 tokens/sec |
| **Best For** | Business logic, APIs, orchestration | Matrix multiplications, LLM inference |
| **Tools for CPU LLM** | llama.cpp, Ollama | vLLM, TGI, DJL DeepSpeed |

---

## Section 6: ML Framework Comparison

| Framework | Type | Best For | Common Use Case |
|---|---|---|---|
| **Scikit-Learn** | Traditional ML | Small tabular data (CSV/SQL) | Simple classification/regression |
| **XGBoost** | Advanced Traditional ML | Complex tabular data | Fraud detection, credit scoring (NatWest) |
| **PyTorch** | Deep Learning engine | Neural networks, LLMs | Fine-tuning, custom model math |
| **TensorFlow** | Deep Learning engine | Neural networks, production at Google scale | Image recognition, Google internal |
| **Hugging Face** | Wrapper on PyTorch | Loading and serving pre-trained LLMs | Model Hub, Inference, PEFT/LoRA |

---

## Section 7: LLM Evaluation Techniques

| Technique | What It Measures | Tools |
|---|---|---|
| **LLM-as-a-Judge** | Grade model output quality using a smarter LLM | GPT-4 as judge, custom prompts |
| **Faithfulness (RAGAS)** | Did model hallucinate? Stick to context? | RAGAS library |
| **Answer Relevance (RAGAS)** | Did model answer the actual question? | RAGAS library |
| **Context Precision** | Did vector DB retrieve the right documents? | RAGAS library |

---

## Section 8: Enterprise Container Strategy (NatWest)

| Step | Action | Tool |
|---|---|---|
| **1. Pull** | Download public vLLM/TGI base image | DockerHub / GitHub Container Registry |
| **2. Scan** | Security vulnerability scan | Prisma Cloud / Twistlock / Aqua |
| **3. Harden** | Add SSL certs, proxy config, logging agents | Custom Dockerfile |
| **4. Push** | Upload to private registry | Amazon ECR / JFrog Artifactory |
| **5. Deploy** | K8s pulls from private ECR only | EKS + KServe |

---

## Section 9: AWS vs Azure Equivalents (Cloud Architecture)

| Concept | AWS Service | Azure Service |
|---|---|---|
| **Model / Adapter Storage** | Amazon S3 | Azure Blob Storage |
| **Container Registry** | Amazon ECR | Azure Container Registry (ACR) |
| **Kubernetes (Orchestration)** | Amazon EKS | Azure Kubernetes Service (AKS) |
| **Managed ML Platform** | Amazon SageMaker | Azure Machine Learning (AML) / **Azure Databricks** |
| **Managed LLM Serving** | SageMaker Endpoints | AML Managed Endpoints / **Databricks Model Serving** |
| **S3 Path Prefix** | `s3://bucket/path` | `az://container/path` or `https://...` |
| **GPU Instances** | `p4d` (A100), `g5` (A10G) | `NC A100 v4` series, `NDm A100 v4` |
| **Model Registry / Tracking** | SageMaker Model Registry | Azure ML Registry / **Databricks MLflow** |
| **Serverless Inference** | SageMaker Serverless | Azure Container Apps / **Databricks Serverless** |

