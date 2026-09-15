"""
HUGGING FACE BOILERPLATE CODE (For MLOps / Platform Engineers)
==============================================================
This is the code you will write 99% of the time on the job.
It uses Hugging Face to do all the heavy lifting for you.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------
# 1. BOILERPLATE: LOAD ANY MODEL (Inference)
# ---------------------------------------------------------
def run_inference(model_id: str, prompt: str):
    print(f"Loading {model_id} via Hugging Face...")
    
    # Load the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load the Model
    # 'device_map="auto"' magically puts it on a GPU if available
    # 'torch_dtype=torch.float16' optimizes memory usage
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Generate Text
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# ---------------------------------------------------------
# 2. BOILERPLATE: APPLY LORA (Fine-Tuning Setup)
# ---------------------------------------------------------
def setup_lora_fine_tuning(model):
    print("Injecting LoRA adapters via Hugging Face PEFT...")
    
    # This dictionary of settings is standard across most fine-tuning jobs
    lora_config = LoraConfig(
        r=8,                                 # Rank (Size of the adapter)
        lora_alpha=16,                       # Scaling factor
        target_modules=["q_proj", "v_proj"], # Target attention layers
        lora_dropout=0.05,                   # Prevents overfitting
        bias="none",
        task_type="CAUSAL_LM"                # Language Modeling
    )
    
    # Wrap the model
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    return peft_model

# Note: After running setup_lora_fine_tuning, you simply pass the `peft_model` 
# into the Hugging Face `Trainer` class and it handles all the math for you!

