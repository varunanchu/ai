# MLOps & GenAI: Command Execution Guide

This document contains all the practical commands we performed (or discussed) to set up and run the end-to-end Local LLM pipeline, complete with descriptions for what each command does.

## 1. Environment & Package Setup

Instead of using Docker/Kubernetes which require high overhead, we set up a "bare-metal" Python environment to host the LLM locally on your CPU.

```powershell
# Install the core Machine Learning and API libraries
pip install torch transformers fastapi uvicorn pydantic huggingface_hub accelerate peft datasets
```
* **`torch` & `transformers`**: The core deep learning engine and Hugging Face library used to load and run the LLM.
* **`fastapi` & `uvicorn`**: Used to build and serve the local REST API.
* **`peft` & `datasets`**: Used specifically for the LoRA fine-tuning process.

## 2. Serving the LLM Locally

We created `server.py` to act as an OpenAI-compatible API serving TinyLlama-1.1B.

```powershell
# Start the Local LLM Server
python server.py
```
* **What it does**: 
  1. Downloads the ~4GB TinyLlama model weights from Hugging Face (first time only).
  2. Loads the model into your laptop's RAM using PyTorch.
  3. Opens a FastAPI web server on `localhost:8000` listening for incoming API requests.

## 3. Testing the Inference API

We created `test_llm_api.py` to simulate an application requesting text generation.

```powershell
# Run the API Test Client
python test_llm_api.py
```
* **What it does**: Sends a POST request (a prompt asking to explain an LLM) to your local `server.py` and measures how long it takes to generate the response on a CPU.

## 4. Evaluating the LLM (LLM-as-a-Judge)

We created `evaluate.py` to automate the testing of the LLM's accuracy and faithfulness.

```powershell
# Run the Evaluation Script
python evaluate.py
```
* **What it does**: Passes a Context, a Question, and a Generated Answer to the LLM, asking it to act as an impartial judge and return a "1" (Pass/Faithful) or "0" (Fail/Hallucination).

## 5. Fine-Tuning the LLM (PEFT / LoRA)

We created `finetune_lora.py` to demonstrate how to efficiently train a large model on limited hardware.

```powershell
# Run the Fine-Tuning Script
python finetune_lora.py
```
* **What it does**:
  1. Loads the base model but *freezes* its weights.
  2. Injects tiny LoRA "adapter" matrices.
  3. Trains *only* the adapters on a mock dataset for 5 epochs.
  4. Saves the newly trained adapter weights to a tiny folder (`./lora_adapter`).

---

## Appendix: Kubernetes Commands (For Reference)
If we were deploying this to AWS EKS or a local Minikube cluster, these are the commands we would have used:

```powershell
# Boot a local Kubernetes cluster
minikube start --driver=docker --memory=3000 --cpus=2

# Deploy the YAML files (Namespace, ConfigMap, Deployment, Service)
kubectl apply -f k8s/

# Check if the pods are running successfully
kubectl get pods -n llm-lab

# Port-forward the Kubernetes service so your local machine can talk to it
kubectl port-forward svc/llm-service 8000:80 -n llm-lab
```

