# MLOps & GenAI: Core Theory & Concepts

This document summarizes the theoretical concepts discussed during our session. These concepts map directly to the enterprise MLOps and GenAI architecture skills on your resume.

---

## 1. PEFT & LoRA (Parameter-Efficient Fine-Tuning)

**The Problem:**
Standard fine-tuning of a Large Language Model (e.g., a 70B parameter model) requires updating every single weight in the network. This takes massive amounts of GPU memory, time, and produces a brand-new 140GB model file for every fine-tuned version.

**The Solution (LoRA - Low-Rank Adaptation):**
Instead of updating the original model, LoRA **freezes** the base model's weights and injects tiny, new matrices (called adapters) into the model's attention layers. You only train these tiny adapters.

*   **Trainable Parameters:** As seen in our script, LoRA reduced the trainable parameters from 1.1 Billion to just 1.1 Million (0.1%).
*   **Cost Savings (~40% GPU inference cost reduction):** In a production environment, you only need to load the heavy Base Model into GPU memory once. You can then dynamically swap the tiny (50MB) LoRA adapters in and out of memory depending on which specific task or tenant is requesting inference. 

## 2. LLM Evaluation (RAGAS & LLM-as-a-Judge)

Evaluating text generation is difficult because you cannot use simple metrics like "accuracy" or "F1 score" on free-flowing text.

**LLM-as-a-Judge:**
Instead of writing complex regex rules, you use a highly capable LLM (like GPT-4) to read the output of your smaller production model and grade it against a specific rubric.
*   *Example:* We prompted the LLM to output exactly a "1" if the answer was good, and a "0" if it was bad.

**RAGAS (Retrieval Augmented Generation Assessment):**
A framework that specifically evaluates RAG (Retrieval-Augmented Generation) pipelines using LLM-as-a-judge. Core metrics include:
*   **Faithfulness:** Did the generated answer stay strictly within the provided context, or did it hallucinate facts?
*   **Answer Relevance:** Did the answer actually address the user's prompt directly?
*   **Context Precision/Recall:** Did the vector database retrieve the correct documents in the first place?

## 3. CPU Inference & Quantization

**The Problem:**
Deep learning models inherently run on GPUs because they require massively parallel matrix multiplications. Running them on CPUs is traditionally too slow and memory-intensive (using Float16 or Float32 data types).

**The Solution (Quantization):**
Quantization compresses the model weights from 16-bit or 32-bit floating-point numbers down to 8-bit or even 4-bit integers (e.g., GGUF formats, bitsandbytes). 
*   *Impact:* A model that requires 16GB of RAM can be compressed to 4GB, allowing it to fit into standard laptop RAM and run efficiently on a CPU, with only a negligible drop in response quality.

## 4. Docker vs. Kubernetes in MLOps

**Docker (The Box):**
Docker creates a sealed "container" that holds your Python environment, your ML libraries (PyTorch, Transformers), and your code. It ensures that the model runs exactly the same on your laptop as it does on AWS.

**Kubernetes (The Warehouse Manager):**
Kubernetes orchestrates those Docker containers across multiple servers (nodes). 
*   **Pod:** The smallest unit in K8s, wrapping your Docker container.
*   **Deployment:** Ensures that exactly $N$ copies of your Pod are always running. If one crashes due to an Out-Of-Memory (OOM) error, K8s automatically restarts it.
*   **Service:** Provides a stable internal IP address/DNS name so other microservices (or your GenAI Gateway) can reliably talk to your LLM pods without worrying about IPs changing.
*   **ConfigMap:** Stores configuration (like `MODEL_NAME=TinyLlama`) outside of the container code, adhering to 12-factor app principles.

