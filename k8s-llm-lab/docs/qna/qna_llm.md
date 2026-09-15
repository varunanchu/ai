# Q&A: MLOps and GenAI Architecture

## Question 1: What libraries have we used till now, and what concepts do they map to?

**Answer:**
Throughout our local lab, we used several key Python libraries. Here is exactly where and why they were used:

1. **`torch` (PyTorch)**
   * **Concept:** Deep Learning Engine / Tensor Computations.
   * **Where used:** Foundation of the entire stack. Both `transformers` and `peft` run on top of PyTorch. It handles moving model weights into RAM/CPU/GPU and calculates the math (gradients) during fine-tuning.

2. **`transformers` (Hugging Face)**
   * **Concept:** Core LLM Loading and Inference.
   * **Where used:** In `server.py` and `finetune_lora.py`. We used `AutoModelForCausalLM` to download and load the Base Model (TinyLlama), `AutoTokenizer` to convert text into numbers (tokens), and the `Trainer` class to manage the training loop.

3. **`peft` (Parameter-Efficient Fine-Tuning)**
   * **Concept:** LoRA (Low-Rank Adaptation) / Optimization.
   * **Where used:** In `finetune_lora.py`. We used `LoraConfig` and `get_peft_model()`. This library is responsible for taking the massive PyTorch model, freezing its core weights, and injecting the tiny trainable adapter matrices into the attention layers.

4. **`datasets` (Hugging Face)**
   * **Concept:** Data Engineering / MLOps Data Pipelines.
   * **Where used:** In `finetune_lora.py`. Used to convert raw Python dictionaries (our mock data) into an optimized, memory-mapped Arrow dataset format that the Hugging Face `Trainer` can ingest efficiently.

5. **`fastapi` & `uvicorn`**
   * **Concept:** LLM Serving / API Gateway.
   * **Where used:** In `server.py`. Used to wrap the Python ML code into a production-grade REST API, allowing external clients to communicate with the model via standard HTTP requests (like OpenAI's API).

6. **`requests` & `pydantic`**
   * **Concept:** API Integration & Data Validation.
   * **Where used:** `pydantic` was used in `server.py` to ensure incoming API payloads matched our strict schema (preventing bad data from crashing the model). `requests` was used in `test_llm_api.py` and `evaluate.py` to act as the client calling the API.

---

## Question 2: Implementing LoRA in a Centralized AWS Hub-and-Spoke Architecture (NatWest Use Case)

**Context:** Centralized AWS account hosting LLMs, serving 100+ different AWS consumer accounts. Can we store adapters in S3 (e.g., different folders for different use cases) and dynamically load them during inference?

**Answer:**
Yes! What you are describing is the absolute gold standard for enterprise LLMOps, known as **Multi-LoRA Serving**. Here is exactly how you architect this on AWS:

### 1. The Storage Layer (S3)
Yes, you store the adapters in S3. Because LoRA adapters are tiny (e.g., 50MB) compared to the base model (e.g., 140GB for Llama 3 70B), S3 is perfect.
Your S3 bucket would look like this:
* `s3://natwest-central-llm/base-models/llama-3-8b/` (Hosted once)
* `s3://natwest-central-llm/adapters/hr-benefits-v1/` (50MB)
* `s3://natwest-central-llm/adapters/finance-fraud-v2/` (50MB)
* `s3://natwest-central-llm/adapters/it-helpdesk-v1/` (50MB)

### 2. The Serving Layer (vLLM or Hugging Face TGI)
To do this in production, you cannot use basic FastAPI. You must use a specialized inference server like **vLLM** or **Hugging Face TGI (Text Generation Inference)** running on Amazon EKS or SageMaker.
These engines natively support **Multi-LoRA**. 

**How it works during inference:**
1. **Startup:** Your centralized EKS pod starts up. It downloads the massive Base Model from S3 and loads it into the GPU VRAM. It *stays* in VRAM permanently.
2. **The API Request:** An application from the HR AWS account sends an API request to your GenAI Gateway. The payload looks like this:
   ```json
   {
       "prompt": "What is the maternity leave policy?",
       "model": "llama-3-8b",
       "lora_name": "hr-benefits-v1"
   }
   ```
3. **Dynamic Loading:** The vLLM server sees the `lora_name`. It pulls the 50MB adapter from the `s3://natwest-central-llm/adapters/hr-benefits-v1/` path (or a local cache on the EKS node).
4. **On-the-Fly Injection:** vLLM injects those 50MB weights into the Base Model *in milliseconds*, processes the HR tokens, and generates the response.
5. **Next Request:** A millisecond later, the Finance account queries the same API. vLLM unloads the HR adapter, injects the Finance adapter, and answers the query. (Advanced engines like vLLM can even process the HR request and Finance request *in the same batch simultaneously* on the same base model!).

### 3. The Business Value (Resume Impact)
If you implemented this at NatWest, you would highlight this on your resume:
* Without Multi-LoRA: Hosting 100 fine-tuned models requires 100 dedicated GPUs. Cost = Millions of dollars.
* With Multi-LoRA: Hosting 100 fine-tuned adapters requires **1 Base Model on 1 cluster of GPUs**. Cost = Reduced by 90%+. 
This perfectly justifies your resume bullet point: *"cutting GPU inference cost by ~40% while maintaining accuracy."*


## Question 3: What is the exact difference between Hugging Face and PyTorch in our architecture? (Is Hugging Face for inference and PyTorch for fine-tuning?)

**Answer:**
Your summary is very close, but there is a slight misconception. **PyTorch is actually doing the heavy lifting for BOTH inference and fine-tuning.**

The easiest way to understand the difference is using a car analogy:
* **PyTorch is the Engine.**
* **Hugging Face is the Steering Wheel and Dashboard.**

### 1. PyTorch (The Mathematical Engine)
PyTorch doesn't know what "English", "Text", or an "LLM" is. It only knows how to multiply massive grids of numbers (tensors) extremely fast using your CPU or GPU.
* **In Fine-Tuning:** PyTorch calculates the complex calculus (gradients) needed to update the model's weights.
* **In Inference (Running the model):** PyTorch does the billions of matrix multiplications required to predict the next word.

### 2. Hugging Face (The Interface / Wrapper)
Hugging Face is a library built **on top** of PyTorch. Writing raw PyTorch code to build an LLM from scratch takes thousands of lines of code. Hugging Face does all the boilerplate work for you.
* It provides the **Tokenizer**: translating human text ("Hello") into the numbers PyTorch needs ([15496]).
* It handles the **Model Hub**: downloading the pre-trained weights from the internet.
* It wraps complex PyTorch operations into simple commands like model.generate().

**To summarize in our architecture:**
Whenever you call Hugging Face (	ransformers or peft) in our scripts, Hugging Face is secretly translating your request and handing it down to PyTorch to do the actual math, whether that math is for generating a response (inference) or updating an adapter (fine-tuning).


## Question 4: I am an intermediate Python coder. How do I handle requests to write Hugging Face or PyTorch code? Can I get universal boilerplate code?

**Answer:**
**Yes, you can absolutely handle this.** Here is the biggest secret in MLOps and GenAI Engineering: **Almost nobody writes raw PyTorch code from scratch.**

Writing raw PyTorch code (defining mathematical layers, matrix dimensions, backpropagation loops) is the job of an *AI Researcher or Data Scientist*. 

Your job as a *GenAI Platform Engineer / MLOps Engineer* is to take existing models and put them into production. For this, you use **Hugging Face**, which is heavily standardized. If you understand dictionaries, functions, and basic object-oriented Python, you have all the skills you need.

Here are the two universal boilerplate templates that you can copy, paste, and use for almost **any** Open-Source model (Llama 3, Mistral, Qwen, etc.).

### Boilerplate 1: Standard Inference (Loading and Running a Model)
If someone asks you: *"Write a script to load a model and generate text."*

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Define the model you want from the Hugging Face Hub
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 2. Load Tokenizer (translates text to numbers)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 3. Load Model (The core boilerplate)
# 'device_map="auto"' magically finds your GPU if you have one, or uses CPU if you don't.
# 'torch_dtype=torch.float16' optimizes memory usage.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto", 
    torch_dtype=torch.float16 
)

# 4. Generate Text
prompt = "Explain MLOps in one sentence:"
# Convert text to PyTorch tensors and send to GPU/CPU
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Tell the model to generate the response
outputs = model.generate(**inputs, max_new_tokens=50)

# Decode the numbers back into English text
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response_text)
```

### Boilerplate 2: LoRA Fine-Tuning Setup
If someone asks you: *"Write the code to apply LoRA to a model for fine-tuning."*

```python
from peft import LoraConfig, get_peft_model

# 1. You assume the 'model' is already loaded using the boilerplate above.

# 2. Define the LoRA Configuration
# This dictionary of settings is the exact same across 99% of fine-tuning jobs.
lora_config = LoraConfig(
    r=8,                                   # Rank (How big the adapter is)
    lora_alpha=16,                         # Scaling factor
    target_modules=["q_proj", "v_proj"],   # Which attention layers to target
    lora_dropout=0.05,                     # Prevents overfitting
    bias="none",
    task_type="CAUSAL_LM"                  # We are training a language model
)

# 3. Inject the LoRA adapters into the base model
peft_model = get_peft_model(model, lora_config)

# Print out how many parameters you are actually training (usually < 1%)
peft_model.print_trainable_parameters()

# From here, you just pass `peft_model` into the Hugging Face `Trainer`!
```


## Question 5: What are the differences between the LLM Serving Engines (TGI, DJL DeepSpeed, vLLM, KServe) and Optimization Techniques (Quantization, Multi-GPU)?

**Answer:**
When you move out of local testing and into AWS/Production, FastAPI isn't fast enough. You need specialized C++/CUDA serving engines.

### The Serving Engines:
1. **vLLM:** The most popular open-source serving engine right now. It is famous for **PagedAttention**. (Imagine how computers use "page memory" for RAM; vLLM does this for GPU memory, allowing you to serve 3x to 4x more users simultaneously without crashing the GPU).
2. **Hugging Face TGI (Text Generation Inference):** The enterprise competitor to vLLM. Built by Hugging Face (in Rust and Python), it natively supports deploying **Multi-LoRA** adapters and is heavily optimized for AWS SageMaker.
3. **DJL DeepSpeed (AWS):** Deep Learning Java Library (DJL) combined with Microsoft's DeepSpeed. This is primarily used on AWS SageMaker when a model is **so massive** (like a 100B parameter model) that it cannot fit on one GPU, and must be perfectly split across 8 GPUs.
4. **KServe:** A Kubernetes-native orchestration tool. It doesn't run the model itself; rather, it *manages* vLLM or TGI containers on Kubernetes, providing auto-scaling (scaling GPUs from 0 to 10 based on web traffic).

### Inference Optimizations:
1. **Quantization (INT8/FP16):** 
   * **FP16 (16-bit Float):** The standard format. A 7B parameter model takes ~14GB of GPU RAM.
   * **INT8 (8-bit Integer):** Compresses the model by cutting the precision in half. The 7B model now takes ~7GB of RAM. It saves you thousands of dollars in AWS costs because you can rent a smaller GPU, with almost no loss in AI smarts.
2. **Distributed/Multi-GPU Inference (Tensor Parallelism):** An A100 GPU has 80GB of RAM. Llama-3-70B requires ~140GB of RAM. You physically cannot load it on one GPU. Multi-GPU inference splits the neural network exactly in half—GPU #1 calculates the left half of the math, GPU #2 calculates the right half, and they communicate via ultra-fast cables (NVLink) to generate the final word.

---

## Question 6: What is the difference between Scikit-learn, XGBoost, and TensorFlow/PyTorch? Why are they used?

**Answer:**
Before LLMs (Generative AI), the data science world was dominated by **Predictive AI**. You use different tools depending on the type of data you have.

### 1. Scikit-learn (Traditional Machine Learning)
* **What it is:** The basic toolkit for standard ML.
* **Why it's used:** Used for **Tabular Data** (Excel spreadsheets, SQL tables). It is used for linear regression (predicting house prices) or clustering (grouping customers). It runs on CPUs and is very lightweight.

### 2. XGBoost (Extreme Gradient Boosting)
* **What it is:** An advanced algorithm based on "Decision Trees". 
* **Why it's used:** It is the undisputed king of **Tabular Data / Financial Data**. If NatWest wants to predict **Credit Card Fraud** or **Customer Churn**, they use XGBoost. It is much more accurate than Scikit-learn for complex spreadsheets, but doesn't do text or images.

### 3. TensorFlow & PyTorch (Deep Learning)
* **What it is:** Frameworks that simulate Neural Networks (mimicking the human brain).
* **Why it's used:** Used for **Unstructured Data** (Images, Audio, Natural Language). If you want to build an LLM, recognize faces in a camera, or translate English to French, you must use Neural Networks. PyTorch has largely won the AI war (used by OpenAI, Meta, Hugging Face), while TensorFlow is heavily used internally at Google.


## Question 7: What is the actual difference between CPU and GPU machines? Why are GPUs preferred for LLMs?

**Answer:**
To understand this, use this analogy:
* **CPU (Central Processing Unit):** A CPU is like a Brilliant Mathematics Professor. Your laptop's Intel i5 has 2 to 4 "cores" (professors). They are incredibly smart and can handle highly complex, sequential logic (like running an operating system, handling databases, or rendering a web page). 
* **GPU (Graphics Processing Unit):** A GPU is like an army of 10,000 middle-school math students. They aren't as smart as the professor, but they can do thousands of simple multiplications *at the exact same time*.

**Why LLMs need GPUs:**
An LLM is essentially a gigantic grid of numbers (matrices). When you ask an LLM a question, it doesn't "think" in logic; it performs billions of simple multiplications (Matrix Multiplication) to predict the next word. 
* If you give this to the 4 CPUs, they have to do the billions of multiplications one by one. (This is why your TinyLlama took 42 seconds to answer yesterday).
* If you give this to an Nvidia A100 GPU, its 6,912 CUDA cores perform all the multiplications simultaneously in parallel. (This generates the answer in 0.1 seconds).

Additionally, GPUs have **VRAM (Video RAM / High Bandwidth Memory)**. Standard laptop RAM transfers data at ~50 GB/second. GPU VRAM transfers data at ~2,000 GB/second. To generate text fast, you need to push the massive model weights into the processor instantly, which only VRAM can do.

---

## Question 8: Can we practically implement enterprise serving engines (vLLM, TGI, DeepSpeed) on a local laptop CPU?

**Answer:**
**The short answer:** No, not directly. 

**The technical reason:**
Frameworks like **vLLM** and **Hugging Face TGI** are fundamentally built on top of **CUDA** (Nvidia's proprietary software layer for GPUs). Their biggest selling points—like **PagedAttention** (which optimizes GPU VRAM blocks)—physically require GPU hardware to function. If you try to run a vLLM Docker container on an Intel i5 laptop, it will crash immediately with a `CUDA not found` error.

**How MLOps Engineers handle local development:**
Instead of running vLLM locally, MLOps engineers do one of two things:
1. **Use a CPU-optimized stand-in:** For local testing, we use `llama.cpp` or `Ollama`. These are written in C/C++ specifically to optimize matrix multiplications for laptop CPUs (using AVX2 instructions instead of CUDA). 
2. **Write the Deployment Code Locally, Run Remotely:** We write the Kubernetes YAML files and Docker commands for vLLM locally on the laptop, but we push that code to an AWS EKS cluster that actually has the Nvidia GPUs attached to it.


## Question 9: Do we use public vLLM/TGI images, or build custom enterprise containers? (NatWest context)

**Answer:**
In a highly regulated bank like NatWest, **you never deploy public DockerHub images directly to production.** 

**The Enterprise Workflow:**
1. **Pull & Scan:** You pull the open-source `vllm/vllm-openai` image. Security tools (like Twistlock, Prisma Cloud, or Aqua) scan it for CVEs (vulnerabilities).
2. **Customize (The Dockerfile):** You build a custom image *on top* of the open-source one. You inject NatWest's internal root SSL certificates, add corporate proxy settings, and embed observability agents (like Splunk or Datadog agents for logging).
3. **Private Registry:** You push this hardened, custom image to a private registry (like Amazon ECR or JFrog Artifactory). 
4. **Deploy:** Kubernetes/SageMaker only pulls images from this secure internal ECR.

---

## Question 10: How does vLLM's "PagedAttention" actually achieve 3x more users under the hood?

**Answer:**
To understand this, you must understand the **KV Cache (Key-Value Cache)**. When an LLM generates a word, it stores the mathematical context of all previous words in GPU VRAM so it doesn't have to recalculate them. This is the KV Cache.

* **The Old Way (Massive Waste):** Before vLLM, if a user requested an output of up to 2048 tokens, the serving engine would instantly pre-allocate a huge, contiguous block of VRAM for 2048 tokens. If the user's answer only ended up being 50 tokens, the other 1998 tokens of VRAM were locked and wasted (this is called *memory fragmentation*). Because VRAM was wasted, the GPU could only handle ~10 users at once before running out of memory.
* **The vLLM Way (PagedAttention):** Inspired by operating systems, vLLM breaks the KV cache into tiny "pages" (blocks of 16 tokens). It allocates memory dynamically, *exactly when it is needed*, instead of pre-allocating a huge chunk upfront. 
* **The Benefit:** Memory waste drops from ~60% to < 4%. Because you freed up 60% of your VRAM, you can now fit 30 users (batches) into the exact same GPU simultaneously. This increases throughput by 3x.

---

## Question 11: Multi-LoRA Architecture Flow - How does it pick adapters dynamically?

**Answer:**
You do **not** need the base model and adapters in the same S3 bucket. Here is the exact architectural flow for serving 100+ use cases (like `uc50`) on one TGI cluster:

1. **Initialization:** The TGI server boots up on your GPU. It downloads the Base Llama-3 model from `s3://base-models/` and locks it permanently into the GPU VRAM.
2. **The Request:** A user from use case 50 makes an API request to your GenAI Gateway. The Gateway routes the request to TGI with a payload: `{"prompt": "Hello", "adapter_id": "s3://adapters/uc50"}`.
3. **The LRU Cache:** TGI looks at the `adapter_id`. It maintains a small **LRU (Least Recently Used) Cache** in VRAM. If `uc50` isn't in VRAM, it instantly fetches the tiny 50MB file from S3 (or a local SSD cache) and loads it into a small dedicated section of VRAM.
4. **On-the-Fly Fusion (The Magic):** During the mathematical "forward pass" (generating the word), TGI computes the math using the Base Model weights, *adds* the math from the `uc50` adapter weights, and generates the specialized token. 
5. **Context Switching:** If the next request asks for `uc99`, TGI swaps `uc99` into the compute kernel. The base model never moves; only the tiny 50MB adapters are swapped in and out of the active compute kernel in milliseconds.

---

## Question 12: DJL DeepSpeed - What are the benefits and how does the architecture flow on SageMaker?

**Answer:**
**The Benefit (Why use it?):** 
A Llama-3 70B model requires ~140GB of VRAM in 16-bit float. The most powerful enterprise GPU (Nvidia A100) only holds 80GB. You **physically cannot** load this model onto a single GPU. You must use a technique called **Tensor Parallelism**, and Microsoft DeepSpeed is the industry standard for doing this.

**The Architectural Flow (SageMaker):**
1. **Provisioning:** You tell SageMaker to create an endpoint using an instance with 4 GPUs (e.g., `ml.g5.12xlarge`). 
2. **DJL Startup:** SageMaker pulls the DJL DeepSpeed Docker container. The container reads your `serving.properties` file where you defined `tensor_parallel_degree=4`.
3. **Model Sharding:** DeepSpeed downloads the 140GB model. Instead of putting it on one GPU, it mathematically slices the neural network vertically. It puts 35GB of the model on GPU 1, 35GB on GPU 2, 35GB on GPU 3, and 35GB on GPU 4.
4. **Inference (All-Reduce):** When a user asks a question, the input goes to all 4 GPUs simultaneously. GPU 1 calculates its 25% of the math, GPU 2 calculates its 25%, etc. 
5. **NVLink Synchronization:** The GPUs communicate with each other over ultra-fast physical cables (NVLink) using an operation called `All-Reduce` to combine their partial answers into the final word. DJL manages this massive distributed orchestra so the user just sees a normal API response.


## Question 13: Summary Check - vLLM (other benefits?), TGI (how quantization works?), and DJL DeepSpeed (other features?)

**Answer:**
Your high-level summary is spot on! Here are the deeper answers to your specific follow-up questions:

### 1. vLLM: Other benefits besides PagedAttention?
While PagedAttention (KV-Cache optimization) is its claim to fame, vLLM has two other massive benefits:
* **Continuous Batching:** Older engines waited for a batch of 10 users to *all* finish generating before taking the next 10 users. vLLM uses continuous batching—the exact millisecond User A finishes their sentence, vLLM ejects them and instantly injects User 11 into the GPU cycle without waiting for the others.
* **OpenAI-Compatible Server:** It comes out-of-the-box with an API that perfectly mimics OpenAI's API. This means if an enterprise app was written to use GPT-4, you can point it at your vLLM server with zero code changes.

### 2. TGI: How exactly is quantization achieved?
TGI natively bundles quantization libraries like `bitsandbytes`, `AWQ`, and `GPTQ`.
* **How it works:** You pass a flag like `--quantize bitsandbytes` when starting the Docker container. 
* **The Math:** When TGI pulls the 16-bit decimal weights (FP16) from S3, it mathematically compresses them into 8-bit or 4-bit integers *as it loads them into the GPU VRAM*. 
* **The Hardware Trick:** Normally, GPUs are bad at doing math with 4-bit integers. TGI uses highly specialized custom CUDA kernels that decompress the 4-bit numbers back to 16-bit *inside the GPU register for a fraction of a millisecond*, does the math, and throws it away. You get 16-bit accuracy while only taking up 4-bit memory space!

### 3. DJL DeepSpeed: Other features besides Tensor Parallelism?
Besides splitting huge models perfectly across 8 GPUs (Tensor Parallelism), DeepSpeed offers:
* **Pipeline Parallelism:** Instead of splitting the model vertically, it splits it horizontally (e.g., Layers 1-20 on GPU 1, Layers 21-40 on GPU 2).
* **ZeRO (Zero Redundancy Optimizer):** This is massive for **Training/Fine-Tuning**. When fine-tuning, the optimizer states (memory overhead) are huge. ZeRO shards the optimizer memory across all GPUs, completely eliminating memory redundancy. 
* **AWS Native Integration:** Since DJL (Deep Learning Java Library) is built by AWS, it has first-class, seamless integration with SageMaker, making it the default choice for deploying massive models on AWS enterprise architecture.


## Question 14: Deep Dive into Multi-LoRA Architecture — End-to-End Flow

**Answer:**

### The Problem Multi-LoRA Solves
Imagine NatWest has 5 business units: HR, Finance, Fraud, IT Helpdesk, and Customer Service. Each unit wants a fine-tuned LLM that speaks their domain language. Without Multi-LoRA:
- Fine-tune 5 separate copies of Llama-3-8B (each = 16GB of VRAM)
- Deploy 5 separate GPU pods
- Pay for 5 GPU machines running 24/7

**Cost = 5x.** Multi-LoRA reduces this to **1 Base Model + 5 tiny adapters = 1 GPU machine.**

---

### The Architecture (Layer by Layer)

#### Layer 1: The Storage Layer (S3)
```
S3 Layout (Model and adapters can be in different buckets!)
─────────────────────────────────────────────────────────
s3://nw-base-models/
    └── llama-3-8b/          ← 16GB base model weights (downloaded ONCE at startup)

s3://nw-lora-adapters/
    ├── hr-benefits/          ← 50MB LoRA adapter (trained on HR policy docs)
    ├── finance-fraud/        ← 50MB LoRA adapter (trained on fraud patterns)
    ├── it-helpdesk/          ← 50MB LoRA adapter (trained on IT runbooks)
    └── customer-service/     ← 50MB LoRA adapter (trained on customer FAQs)
```
Important: The model and adapters are in DIFFERENT S3 paths. The serving engine knows both.

#### Layer 2: The Serving Engine (TGI on EKS/SageMaker)
**Startup Sequence (happens once when the pod boots):**
1. TGI container starts on the GPU machine.
2. It downloads the 16GB base model from `s3://nw-base-models/llama-3-8b/`
3. It loads the base model permanently into GPU VRAM. It never leaves.
4. TGI registers the adapter paths from config: `"adapter_ids": ["hr-benefits", "finance-fraud", ...]`
5. It maintains a small LRU (Least Recently Used) adapter cache in VRAM (e.g., holds last 4 used adapters)

#### Layer 3: The GenAI Gateway (Your Routing Layer)
The GenAI Gateway is the single endpoint that the 100+ AWS accounts hit. Based on the account or use-case tag, the Gateway adds the `adapter_id` to the payload before forwarding to TGI.

```
Consumer AWS Accounts            GenAI Gateway              TGI on GPU
─────────────────────            ─────────────              ────────────
HR Account sends:          →     Gateway tags it:      →    TGI receives:
{ "prompt": "maternity" }        { "prompt": "...",         { "prompt": "...",
                                   "adapter_id": "hr-benefits" }  "adapter_id": "hr-benefits" }
```

#### Layer 4: The Inference Engine (The Magic Inside TGI)
When TGI receives the request with `adapter_id: hr-benefits`:

**Step 1 - Adapter Lookup:**
```
Is "hr-benefits" already in my VRAM LRU Cache?
    YES → Use it directly (0ms overhead)
    NO  → Download 50MB from S3 (takes ~100ms), add to VRAM cache
```

**Step 2 - Mathematical Fusion:**
In every neural network layer, TGI does TWO sets of math simultaneously:
```
Standard Output = (Base Weight Matrix) × (Input Tensor)  ← Always runs
Adapter Output  = (Adapter A matrix) × (Adapter B matrix) × (Input Tensor)  ← Per-adapter
Final Output    = Standard Output + (lora_alpha / r) × Adapter Output
```
The final token is the *sum* of the base model's output and the adapter's output. This is why an HR fine-tuned model still has general English ability (from the base) but also knows NatWest maternity leave policy (from the adapter).

**Step 3 - LRU Cache Eviction:**
If the next request comes in for `finance-fraud` and the VRAM cache is full:
- TGI evicts the least recently used adapter from VRAM (e.g., `it-helpdesk` if nobody used IT for 5 minutes)
- Downloads `finance-fraud` adapter from S3 into the freed VRAM slot
- Answers the Finance query

#### Layer 5: The Result

```
Timeline (All on 1 GPU Machine)
────────────────────────────────────────────────
T=0ms   → HR request arrives    → Uses HR adapter
T=5ms   → Finance request arrives → Swaps to Finance adapter (or runs both in batch!)
T=10ms  → IT request arrives    → Uses IT adapter (already in cache)
T=15ms  → Customer svc arrives  → Uses CS adapter

One base model. 4 adapters. Millisecond switching. Zero wasted GPUs.
```

---

### The Docker/TGI Command to Enable Multi-LoRA
```bash
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Meta-Llama-3-8B \
  --enable-lora \                              # Enable the Multi-LoRA engine
  --lora-ids hr-benefits,finance-fraud,it-helpdesk \   # Register all adapters
  --max-lora-rank 64 \                         # Max allowed LoRA rank
  --lora-extra-vocab-size 256
```

### The API Call with Adapter Specified (Python)
```python
import requests

# HR team calling the same endpoint but with their specific adapter
response = requests.post("http://nw-genai-gateway/generate",
    json={
        "inputs": "What is NatWest's maternity leave policy?",
        "parameters": {
            "adapter_id": "hr-benefits",   # THIS is how the model knows which adapter!
            "max_new_tokens": 200
        }
    }
)
```


## Question 15: How does adapter_id resolve to a full S3 path? And how does Multi-LoRA handle 100+ concurrent use cases?

---

### Part 1: How does adapter_id resolve to the actual S3 path?

There are THREE approaches used in production. Each one is used depending on your architecture:

---

**Approach A: The adapter_id IS the full S3 path (Simplest)**
The most straightforward approach. Your GenAI Gateway does not use a short name at all.
The API payload contains the complete S3 path as the adapter_id:

```python
# The Gateway sends the FULL S3 path directly
response = requests.post("http://tgi-server/generate",
    json={
        "inputs": "What is our maternity leave policy?",
        "parameters": {
            "adapter_id": "s3://nw-lora-adapters/hr-benefits/",  # Full path!
            "max_new_tokens": 200
        }
    }
)
```
TGI knows exactly where to download from. No lookup needed.

---

**Approach B: Pre-Registered Adapter Map at TGI Startup (Most Common in Enterprise)**
When TGI starts up, you provide a COMPLETE mapping of short names to S3 paths.
TGI builds an internal registry dictionary from this config:

```bash
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Meta-Llama-3-8B \
  --enable-lora \
  --lora-ids \
      "hr-benefits=s3://nw-lora-adapters/hr-benefits/" \      # short name → S3 path
      "finance-fraud=s3://nw-lora-adapters/finance-fraud/" \
      "it-helpdesk=s3://nw-lora-adapters/it-helpdesk/"
```

TGI internally stores this dictionary:
```
{
  "hr-benefits":    "s3://nw-lora-adapters/hr-benefits/",
  "finance-fraud":  "s3://nw-lora-adapters/finance-fraud/",
  "it-helpdesk":    "s3://nw-lora-adapters/it-helpdesk/"
}
```
When a request arrives with `adapter_id: "hr-benefits"`, TGI looks it up in its internal
registry dictionary and resolves the full S3 path instantly in memory.

---

**Approach C: Gateway-Level Resolution via a Config Database (NatWest-Scale)**
At NatWest scale, managing 100+ adapter mappings inside a Docker command is messy.
The cleaner enterprise pattern uses a centralized config store (AWS DynamoDB, SSM, or Redis):

```
Consumer Call           GenAI Gateway                 DynamoDB              TGI Server
─────────────           ─────────────                 ────────              ──────────
{ prompt: "...",   →    1. Extract use_case_id  →     Lookup:           →   Receives full
  use_case: "uc50" }    2. Query DynamoDB:            uc50 →               S3 path in
                           GET /adapters/uc50          s3://nw-lora-         payload
                        3. Get full S3 path            adapters/uc50/
                        4. Attach to TGI request
```
This way, adding a new use case (uc101) is just a DynamoDB entry update, no TGI restart needed.

---

### Part 2: What if all 100+ use cases try to do inference simultaneously?

This is the most important architectural question. The answer is:
**You CANNOT serve 100+ different adapters simultaneously from a SINGLE TGI pod.**

Here is exactly what happens and how it is solved:

---

**The VRAM Math (Why one pod is not enough):**
```
GPU VRAM Available:         80GB (Nvidia A100)
Base Llama-3-8B Model:    - 16GB (permanent, never leaves)
Available for adapters:     64GB remaining

Each LoRA adapter size:     ~50MB
Adapters that fit in VRAM:  64,000MB / 50MB = ~1,280 adapters can fit!
```
Wait — so actually 1,280 adapters CAN fit in VRAM? Theoretically yes, but this is not the real bottleneck.

**The REAL bottleneck: Compute Throughput (NOT VRAM)**
The GPU's compute cores can only run a certain number of parallel batches per second.
When all 100 teams hit the API simultaneously, requests are queued. The issue is:
- Each request waits for compute time, not VRAM space.
- This causes LATENCY (slow response times) even if VRAM is fine.

---

**The Solution: Horizontal Scaling + Smart Batching**

**Strategy 1 — Continuous Batching (Handle multiple adapters in one forward pass)**
TGI does NOT process one request at a time. It groups requests together:

```
Request Queue at one moment:
  - Request 1: adapter=hr-benefits, prompt="maternity leave?"
  - Request 2: adapter=finance-fraud, prompt="flag this transaction"
  - Request 3: adapter=hr-benefits, prompt="sick leave policy?"
  - Request 4: adapter=it-helpdesk, prompt="reset my password"

TGI Batching Engine groups by adapter:
  Batch A (hr-benefits adapter active):    [Request 1, Request 3]  → runs together
  Batch B (finance-fraud adapter active):  [Request 2]             → runs next
  Batch C (it-helpdesk adapter active):    [Request 4]             → runs next
```
Requests using the SAME adapter are batched together to minimize adapter swapping.

**Strategy 2 — Horizontal Scaling with KServe (The Real Answer)**
When 100+ teams ALL need simultaneous low-latency inference, you scale HORIZONTALLY.
You run MULTIPLE TGI pods, not one:

```
                        NatWest GenAI Gateway (Load Balancer)
                                      |
              ┌───────────────────────┼───────────────────────┐
              |                       |                       |
     TGI Pod 1 (GPU A)       TGI Pod 2 (GPU B)       TGI Pod 3 (GPU C)
     Llama-3-8B loaded        Llama-3-8B loaded       Llama-3-8B loaded
     Serves: hr, finance       Serves: fraud, IT       Serves: cs, legal
              |                       |                       |
     Handles 30 req/sec       Handles 30 req/sec      Handles 30 req/sec

Total Throughput: 90 requests/second across 100+ use cases
```

KServe autoscales this automatically:
```yaml
spec:
  predictor:
    minReplicas: 2    # Always keep at least 2 GPU pods running
    maxReplicas: 10   # Scale up to 10 GPU pods during peak hours
    scaleTarget: 30   # Scale up when avg requests per pod exceeds 30/sec
```

**Strategy 3 — Hot vs Cold Adapters (LRU Cache Optimization)**
Not all 100 adapters are equally popular. In any bank, a few teams generate 80% of traffic.
Configure TGI to permanently keep the top adapters "hot" in VRAM:

```bash
--lora-ids "hr-benefits=s3://...;always-hot=true" \  # Never evict from VRAM
           "finance-fraud=s3://...;always-hot=true" \ # Never evict
           "it-uc73=s3://...;on-demand=true"           # Load on first request, evict if idle
```

---

**Summary: Supporting ALL 100+ Use Cases Simultaneously**
```
Requirement                   Solution
──────────────────            ─────────────────────────────────────────────
S3 path resolution            Pre-registered map in TGI config / DynamoDB
VRAM for adapters             LRU cache: hot adapters permanent, cold = on-demand
Concurrent 100+ teams         Horizontal scaling: 3-10 TGI pods via KServe autoscaler
Minimizing adapter swaps      Continuous batching: group same-adapter requests together
New use case onboarding       Just add a DynamoDB entry + upload adapter to S3. Zero TGI restart.
```


## Question 16: Complete clarity on adapter_id → S3 path resolution, and KServe multi-pod architecture

---

### Part 1: The COMPLETE picture of adapter_id → S3 path (step by step)

Let me clear this up completely. The confusion is about WHERE the mapping lives and WHO does the lookup.

---

#### Step 1: You register the mapping ONCE at TGI startup

When you start TGI with the --lora-ids flag, TGI reads this ONCE and builds
an internal Python dictionary in its own memory (not GPU VRAM, just RAM):

```bash
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Meta-Llama-3-8B \
  --enable-lora \
  --lora-ids "hr-benefits=s3://nw-lora-adapters/hr-benefits/"
             "finance-fraud=s3://nw-lora-adapters/finance-fraud/"
```

Internally, TGI now stores:
```python
# This is inside TGI's own code. You did NOT write this. TGI built it from --lora-ids flag.
ADAPTER_REGISTRY = {
    "hr-benefits":    "s3://nw-lora-adapters/hr-benefits/",
    "finance-fraud":  "s3://nw-lora-adapters/finance-fraud/",
}
```

---

#### Step 2: At request time, TGI does the lookup INTERNALLY (no custom code needed)

```
API Request arrives:
{ "inputs": "maternity leave?", "adapter_id": "hr-benefits" }
                    |
                    ↓
TGI's internal router:
  1. Gets adapter_id = "hr-benefits"
  2. Looks up ADAPTER_REGISTRY["hr-benefits"]
  3. Gets → "s3://nw-lora-adapters/hr-benefits/"
  4. Checks LRU cache: Is this adapter already in GPU VRAM?
       YES → Skip download, use it
       NO  → Download from S3 using boto3 (AWS SDK built into TGI)
  5. Runs inference with base model + hr-benefits adapter
```

You do NOT write any custom Python code for this basic flow.
TGI handles everything internally — the dictionary, the S3 download, the caching.

---

#### Step 3: What if you want to ADD a new adapter WITHOUT restarting TGI?

This is where custom code comes in. TGI provides a REST admin API:

```python
import requests

# Call TGI's built-in admin endpoint to register a new adapter on the fly
# This adds "uc-101" to TGI's internal registry WITHOUT restarting the container
response = requests.post("http://tgi-server:8080/adapters/load",
    json={
        "adapter_id": "uc-101",
        "adapter_source": "s3",
        "adapter_path": "s3://nw-lora-adapters/uc-101/"
    }
)
# From this moment, any request with adapter_id: "uc-101" will work
```

This is the custom code you would write in your GenAI Gateway microservice
when a new use case team is onboarded. Just one API call. No restart.

---

#### How does TGI actually READ from S3? (The technical mechanism)

TGI uses the huggingface_hub library + boto3 (AWS SDK) internally.
You provide the AWS credentials to the TGI container via environment variables:

```bash
docker run --gpus all -p 8080:80 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_DEFAULT_REGION=eu-west-2 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id s3://nw-base-models/llama-3-8b/ \
  --enable-lora
```

TGI authenticates to S3 using those credentials.
When it sees an S3 path, it calls the AWS S3 API to stream-download
the adapter files directly into host RAM, then copies to GPU VRAM.

---

#### Running this on Azure instead of AWS

On Azure, instead of S3, you use Azure Blob Storage.
The paths change from s3:// to az:// (or https://):

```bash
docker run --gpus all -p 8080:80 \
  -e AZURE_STORAGE_ACCOUNT=nwllamastorage \
  -e AZURE_STORAGE_KEY=your_azure_key \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id "az://nw-base-models/llama-3-8b" \
  --lora-ids "hr-benefits=az://nw-lora-adapters/hr-benefits/"
```

Everything else — the adapter_id lookup, the LRU cache, the inference — 
works identically. The only difference is the storage credential and path prefix.

---

### Part 2: KServe Multi-Pod Architecture — Complete Clarity

The single BIGGEST misconception to clear up:

!!! There is NO "main model pod" that other pods talk to !!!
!!! EACH pod is 100% INDEPENDENT and runs its OWN copy of the base model !!!

---

#### How KServe Actually Works (End to End)

```
INTERNET / Internal NatWest Apps
              |
              ↓
    ┌─────────────────────┐
    │   KServe Ingress    │  ← This is Istio/Envoy (load balancer built into KServe)
    │ (Load Balancer)     │    It distributes requests across pods using Round Robin
    └─────────────────────┘
         /       |       \
        /        |        \
       ↓         ↓         ↓
┌───────────┐ ┌───────────┐ ┌───────────┐
│  TGI Pod 1│ │  TGI Pod 2│ │  TGI Pod 3│  ← 3 completely SEPARATE pods
│           │ │           │ │           │
│ GPU A     │ │ GPU B     │ │ GPU C     │  ← Each has its OWN GPU
│ Llama-3   │ │ Llama-3   │ │ Llama-3   │  ← Each loaded its OWN copy of base model
│ (16GB)    │ │ (16GB)    │ │ (16GB)    │
│           │ │           │ │           │
│ HR adapter│ │ Finance   │ │ IT adapter│  ← Each has its OWN adapter LRU cache
│ cached    │ │ adapter   │ │ cached    │
└───────────┘ └───────────┘ └───────────┘

The pods NEVER talk to each other. 
The load balancer decides which pod gets each request.
There is no coordination between pods at inference time.
```

---

#### How does the load balancer decide which pod to send the request to?

KServe uses Istio (a service mesh) as its internal load balancer.
By default it uses Round Robin (Pod1 → Pod2 → Pod3 → Pod1 ...).

But you can configure it for "Sticky Sessions" per adapter:
- All requests with adapter_id=hr-benefits → always go to Pod 1
- All requests with adapter_id=finance-fraud → always go to Pod 2
This minimizes adapter swapping because each pod specializes in certain adapters.

---

#### Does running multiple pods cost more GPUs? YES — but here is the math:

**WITHOUT Multi-LoRA (Old Way):**
```
100 business units × 1 dedicated fine-tuned model GPU each = 100 GPUs
Monthly cost: 100 × $3,000/month = $300,000/month
```

**WITH Multi-LoRA + KServe (New Way):**
```
3 TGI pods (each with 1 GPU, serving all 100 use cases) = 3 GPUs
But KServe autoscales:
  - During office hours (9am-6pm): scales UP to 8 pods (peak traffic)
  - During night (10pm-7am): scales DOWN to 1 pod (idle)

Monthly cost: roughly 3-8 GPUs average × $3,000 = $9,000-$24,000/month
```

**Saving: $276,000/month. That is the ROI of Multi-LoRA.**

---

#### KServe Autoscaling — How the scaling decision is made:

```yaml
# In your KServe InferenceService YAML
spec:
  predictor:
    minReplicas: 1      # Never go below 1 pod (always available)
    maxReplicas: 10     # Never exceed 10 pods
    scaleMetric: rps    # Scale based on Requests Per Second
    scaleTarget: 30     # If any pod exceeds 30 req/sec, add a new pod
```

The scaling flow:
1. Traffic spikes to 90 req/sec on Pod 1.
2. KServe sees Pod 1 is over its 30 req/sec target.
3. KServe tells Kubernetes: "Launch 2 more GPU pods."
4. Kubernetes provisions 2 more GPU nodes from AWS (ml.g5.xlarge, for example).
5. TGI starts on those nodes, downloads the base model from S3 (takes 2-5 minutes).
6. Load balancer starts routing traffic to the new pods.

The 2-5 minute cold start is a real production issue!
Solution: Use Kubernetes "Warm Pool" — keep a few GPU nodes idle and pre-loaded,
so scaling takes 10 seconds instead of 5 minutes.
