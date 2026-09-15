"""
LLM Evaluation Script (LLM-as-Judge & Faithfulness)
===================================================
This script demonstrates how MLOps pipelines evaluate model outputs.
It uses the local LLM to "grade" an answer based on a given context.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def llm_as_judge(question: str, context: str, generated_answer: str):
    """
    Uses the LLM to grade if the `generated_answer` is faithful to the `context`.
    In production, this prompt is usually sent to GPT-4 or Claude 3.5 Sonnet.
    """
    
    evaluation_prompt = f"""<|system|>
You are an impartial AI evaluator. Your job is to evaluate if the Answer is FAITHFUL to the Context. 
If the answer contains information NOT present in the context, it is a hallucination (Score: 0).
If the answer is completely supported by the context, it is faithful (Score: 1).
Respond ONLY with a 1 or a 0. Do not explain.</s>
<|user|>
Context: {context}
Question: {question}
Answer: {generated_answer}
Score:</s>
<|assistant|>"""

    print("\n" + "="*50)
    print("EVALUATING OUTPUT (Faithfulness Check)")
    print(f"Question : {question}")
    print(f"Answer   : {generated_answer}")
    print("="*50)
    
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/completions",
            json={
                "prompt": evaluation_prompt,
                "max_tokens": 5, # We only need a 1 or 0
                "temperature": 0.1 # Low temperature for strict grading
            },
            timeout=120
        )
        
        if resp.status_code == 200:
            score = resp.json()['text'].strip()
            print(f"Judge's Score: {score}")
            if "1" in score:
                print("Result: PASS (Faithful to context)")
            else:
                print("Result: FAIL (Hallucination detected)")
        else:
            print("Error connecting to LLM server.")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    context = "Varun P G is a Senior GenAI Platform Engineer at NatWest Group."
    question = "Where does Varun work?"
    
    # Test 1: A good, faithful answer
    good_answer = "Varun works at NatWest Group."
    llm_as_judge(question, context, good_answer)
    
    # Test 2: A hallucinated answer
    bad_answer = "Varun works at Google as a software developer."
    llm_as_judge(question, context, bad_answer)

