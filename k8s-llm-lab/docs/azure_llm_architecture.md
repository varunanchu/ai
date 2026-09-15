# Azure LLM Inferencing & Architecture Guide
**Enterprise GenAI Deployment on Microsoft Azure**

While many open-source tools default to AWS documentation, enterprise banks often run multi-cloud or Azure-exclusive environments. This guide explains how to translate industry-standard LLM architectures directly to Microsoft Azure.

---

## 1. Multi-LoRA Architecture on Azure

The architecture remains identical to AWS, but the managed services change.

### The Storage Layer (Azure Blob Storage)
Instead of Amazon S3, models and adapters are stored in **Azure Blob Storage Containers**.
*   **Base Model:** `az://nw-base-models/llama-3-8b/`
*   **HR Adapter:** `az://nw-lora-adapters/hr-benefits/`

### The Serving Engine (TGI on Azure Kubernetes Service - AKS)
You deploy Hugging Face TGI exactly as you would on EKS, but you authenticate it to Azure.
*   **Container Image:** `ghcr.io/huggingface/text-generation-inference`
*   **Infrastructure:** Azure Kubernetes Service (AKS) with `NC A100 v4` series node pools (Azure's A100 GPUs).

**The Azure Docker Implementation:**
```bash
docker run --gpus all -p 8080:80 \
  -e AZURE_STORAGE_ACCOUNT=nwllamastorage \
  -e AZURE_STORAGE_KEY=your_azure_key \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id "az://nw-base-models/llama-3-8b" \
  --enable-lora \
  --lora-ids "hr=az://nw-lora-adapters/hr-benefits/"
```

### The Routing Layer (Azure API Management)
Instead of an AWS API Gateway, you use **Azure API Management (API-M)**.
When the HR frontend calls the API-M, the API-M inspects the JWT token, determines the user is HR, appends `"adapter_id": "hr"`, and routes it to the internal AKS cluster running TGI.

---

## 2. DJL DeepSpeed on Azure (For Massive 70B+ Models)

**The Catch:** DJL (Deep Learning Java Library) is built by AWS. Because of this, it is a first-class citizen on Amazon SageMaker. 
**The Solution:** However, it is fully open-source. To run DJL DeepSpeed on Azure, you use Azure Machine Learning's (AML) **Bring Your Own Container (BYOC)** feature, or deploy it directly on AKS.

### Implementation on Azure Machine Learning (AML)

1.  **The Container:** You pull the open-source AWS container: `deepjavalibrary/djl-serving:latest`.
2.  **The Azure Endpoint:** You create an **Azure ML Managed Online Endpoint**. You configure the compute instance to be `Standard_ND96amsr_A100_v4` (an Azure VM that physically contains 8 Nvidia A100 GPUs).
3.  **The Configuration:** You package the model weights inside Azure Blob Storage along with the exact same `serving.properties` file used in AWS.

**serving.properties (Uploaded to Azure Blob Storage):**
```properties
engine=DeepSpeed
option.model_id=az://nw-massive-models/llama-3-70b
option.tensor_parallel_degree=8     # DeepSpeed shards model across Azure's 8 GPUs
option.dtype=fp16
```

### Architectural Flow for DJL on Azure:
1.  Azure ML provisions the 8-GPU Endpoint.
2.  Azure ML pulls the DJL container and mounts the Blob Storage container.
3.  DJL starts up, reads the `serving.properties`.
4.  DeepSpeed partitions the 140GB Llama-3-70B model into 8 chunks.
5.  It loads ~17.5GB onto each of the 8 Azure A100 GPUs.
6.  When an Azure API request arrives, it splits the math 8 ways, uses NVLink to combine the result (All-Reduce), and returns the text.

---

## 3. Serverless Inference & Autoscaling on Azure

If you want the autoscaling capabilities of KServe (scaling from 0 to 10 GPUs based on traffic), Azure provides two native ways to do this:

### Option A: Azure Container Apps (Easiest)
Azure Container Apps is a serverless platform built on top of Kubernetes and KEDA (Kubernetes Event-driven Autoscaling).
You can deploy your vLLM or TGI container here, and KEDA will automatically scale the number of GPU replicas up and down based on the HTTP request queue length.

### Option B: KServe on Azure Kubernetes Service (Enterprise)
For a direct 1:1 match with the AWS architecture, you install the open-source KServe operator directly onto your Azure Kubernetes Service (AKS) cluster. The `InferenceService` YAML file you write is 100% identical to the one you write for AWS. KServe interacts with the Azure Load Balancer seamlessly.

---

## Summary of Equivalent Commands

| Action | AWS / SageMaker | Azure / AML |
| :--- | :--- | :--- |
| **Download Adapter** | `aws s3 cp s3://bucket/adapter` | `az storage blob download az://container/adapter` |
| **Create Endpoint** | `aws sagemaker create-endpoint` | `az ml online-endpoint create` |
| **GPU Node Pool** | `p4d.24xlarge` (8x A100) | `Standard_ND96amsr_A100_v4` (8x A100) |
| **Engine Used** | DJL DeepSpeed (Native) | DJL DeepSpeed (Custom Container BYOC) |



## 4. Azure Databricks for GenAI (Mosaic AI & MLflow)

**Azure Databricks** is heavily used in enterprise for GenAI, primarily because Databricks acquired MosaicML. It acts as a unified platform that combines your Data Lake (Delta Tables) with LLMOps. 

If your company uses Azure Databricks, you can bypass managing AKS or Azure ML entirely for many of these tasks.

### A. Fine-Tuning (LoRA / PEFT) on Databricks
Instead of manually provisioning VMs and writing raw PyTorch loops, you use **Databricks Mosaic AI Training**.
*   **The Data Flow:** Your training data is stored securely in Azure Data Lake as Delta Tables.
*   **The Orchestration:** You launch a Databricks cluster with GPUs attached. You run your standard Hugging Face PEFT/LoRA code.
*   **The Tracking:** Databricks inherently uses **MLflow**. As your LoRA adapter trains, every metric (loss, epoch, memory usage) and the final adapter weights are automatically logged and tracked in the MLflow Registry.

### B. LLM Model Serving on Databricks
Databricks offers a fully managed service called **Databricks Model Serving (Provisioned Throughput)**. 
*   **No Infrastructure Management:** You do not need to manually write Kubernetes YAML, configure KServe, or pull vLLM/TGI Docker containers.
*   **How it works:** You take your fine-tuned model from the MLflow Registry and click "Serve" (or use Terraform/REST API). 
*   **The Backend:** Under the hood, Databricks spins up Serverless GPUs on Azure. Databricks internally uses highly optimized serving engines (customized versions of vLLM and NVIDIA Triton) to automatically provide **Continuous Batching** and **PagedAttention** without you having to configure the C++ containers yourself.

### C. LLM Evaluation (RAGAS Equivalent)
Instead of running RAGAS manually in a script, Databricks provides **MLflow LLM Evaluate**, which has built-in LLM-as-a-Judge capabilities to test your model for hallucinations and faithfulness before you deploy it to the Model Serving endpoint.
