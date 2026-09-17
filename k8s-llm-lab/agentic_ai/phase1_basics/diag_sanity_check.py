"""
Quick sanity check: Does the model generate coherent text at all?
Run this before running the full agent scripts.
"""
import ollama

MODEL = "llama3.2:1b"

print(f"Testing {MODEL}...")
print("Test 1: Simple hello")
r = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Say exactly: Hello, I am working!"}],
    options={"num_predict": 20, "temperature": 0}
)
print(f"  Response: {r.message.content}")

print("\nTest 2: Simple math")
r = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "What is 5 + 3? Reply with just the number."}],
    options={"num_predict": 10, "temperature": 0}
)
print(f"  Response: {r.message.content}")

print("\nTest 3: Follow instructions")
r = ollama.chat(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You must reply with exactly: Action: calculator"},
        {"role": "user", "content": "What tool should I use to calculate 5+3?"}
    ],
    options={"num_predict": 30, "temperature": 0}
)
print(f"  Response: {r.message.content}")
print("\nSanity check complete!")

