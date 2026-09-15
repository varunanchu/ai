"""
Local Fine-Tuning Script (LoRA)
===============================
This script demonstrates how to fine-tune TinyLlama on a custom dataset
using PEFT (Parameter-Efficient Fine-Tuning) and LoRA (Low-Rank Adaptation).

Why LoRA?
Instead of updating 1.1 billion parameters (which would crash your laptop),
LoRA freezes the model and only trains tiny "adapter" matrices.
"""

import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. Setup Model and Tokenizer
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading Base Model: {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token # Needed for batching

# Load model on CPU (float32 is safe for all CPUs)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map="cpu", 
    torch_dtype=torch.float32 
)

# 2. Apply LoRA Adapters
print("Injecting LoRA Adapters...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,               # Rank of the adapter (higher = more capacity, slower training)
    lora_alpha=16,     # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"] # Target attention layers
)

# This wraps the original model in the PEFT framework
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters() 
# You will see it only trains ~0.1% of the total parameters!

# 3. Create a Tiny Custom Dataset
# In a real pipeline, you would load this from S3 or HuggingFace Datasets
print("Preparing Dataset...")
custom_data = {
    "text": [
        "<|user|>\nWhat is your name?</s>\n<|assistant|>\nI am an AI assistant built by Varun!</s>",
        "<|user|>\nWho created you?</s>\n<|assistant|>\nI was fine-tuned by Varun P G, the GenAI Platform Engineer.</s>",
        "<|user|>\nWhat are your core skills?</s>\n<|assistant|>\nI specialize in MLOps, RAG, and Agentic AI, just like my creator Varun.</s>"
    ]
}

dataset = Dataset.from_dict(custom_data)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. Training Configuration
print("Starting Training on CPU (this takes a few minutes)...")
training_args = TrainingArguments(
    output_dir="./lora_adapter",
    per_device_train_batch_size=1,  # Keep batch size 1 for 8GB RAM
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate batch size 4
    learning_rate=2e-4,
    num_train_epochs=5,             # Just 5 epochs for this tiny dataset
    logging_steps=1,
    save_strategy="no",             # Don't save intermediate checkpoints to save disk space
    use_cpu=True,                   # Force CPU training
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 5. Train and Save!
trainer.train()

print("Training Complete! Saving adapter weights...")
peft_model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")
print("SUCCESS Saved LoRA adapter to ./lora_adapter")

