"""
Local LLM Test Client
=====================
This script sends test prompts to your locally hosted LLM API.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_completion(prompt: str, label: str = ""):
    print(f"\n{'='*50}")
    print(f"TEST: {label}")
    print(f"Prompt: {prompt}")
    print("Generating response (this may take a minute on CPU)...")

    start = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 50, # Keep it short for CPU inference
                "temperature": 0.7
            },
            timeout=120
        )
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"\nResponse : {data['text']}")
            print(f"Time taken : {elapsed:.1f}s")
        else:
            print(f"Error {resp.status_code}: {resp.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the server. Is server.py running?")

if __name__ == "__main__":
    print("🚀 Local LLM API Test")
    
    test_completion(
        prompt="Explain what a large language model is in one short sentence.",
        label="Basic Explanation"
    )
