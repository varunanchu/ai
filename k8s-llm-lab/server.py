"""
Local LLM API Server (Pure Python / Hugging Face)
==================================================
This script loads TinyLlama-1.1B using the standard Hugging Face
transformers library and serves it via an OpenAI-compatible API.
"""

import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pipe = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    print(f"[STARTUP] Downloading/Loading Model: {MODEL_ID}")
    print(f"[STARTUP] This will take ~2-3 minutes the first time...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Load model (optimized for CPU with torch.bfloat16 if supported, else float32)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="cpu", 
        torch_dtype=torch.float32 # Safe default for all CPUs
    )
    
    # Create generation pipeline
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device=-1 # Force CPU
    )
    print("[STARTUP] SUCCESS Model loaded successfully! Server is ready.")
    yield
    print("[SHUTDOWN] Shutting down.")


app = FastAPI(title="Local TinyLlama API", lifespan=lifespan)

# Request schema
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7

# Response schema
class CompletionResponse(BaseModel):
    text: str
    model: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/v1/completions", response_model=CompletionResponse)
def complete(req: CompletionRequest):
    print(f"Received prompt: {req.prompt}")
    
    # Generate text
    outputs = pipe(
        req.prompt, 
        max_new_tokens=req.max_tokens, 
        temperature=req.temperature,
        do_sample=True,
        truncation=True
    )
    
    generated_text = outputs[0]["generated_text"]
    
    # Strip the original prompt from the output
    if generated_text.startswith(req.prompt):
        generated_text = generated_text[len(req.prompt):].strip()
        
    return CompletionResponse(
        text=generated_text,
        model=MODEL_ID
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
