"""
PHASE 1 - SCRIPT 1: What IS an Agent? (Custom ReAct Loop)
==========================================================
Demonstrates the KEY DIFFERENCE between:
  A) A regular LLM call (no tools, no loop)
  B) An Agent (LLM + tools + ReAct loop built from scratch)

This custom implementation uses raw ollama.chat() so you can see
EXACTLY what happens under the hood in every agent framework.
"""

import ollama
import re
from datetime import date

MODEL = "llama3.2:1b"

# ─────────────────────────────────────────────────────────────────────────────
# TOOLS: Plain Python functions
# ─────────────────────────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Evaluates a Python math expression."""
    try:
        return str(round(eval(expression), 4))
    except Exception as e:
        return f"Error: {e}"

def get_current_date(input: str = "") -> str:
    """Returns today's date as YYYY-MM-DD."""
    return str(date.today())

TOOLS = {
    "calculator": calculator,
    "get_current_date": get_current_date,
}

# ─────────────────────────────────────────────────────────────────────────────
# THE REACT SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a reasoning agent. Solve problems step by step using tools.

Available tools:
- calculator: Evaluates math like "144 * 99". Returns the exact result.
- get_current_date: Returns today's date. Input is empty string.

You MUST follow this exact format for EVERY response:
Thought: <your reasoning>
Action: <tool name>
Action Input: <input to tool>

When you have enough information to answer, respond with:
Thought: I now have all the information needed.
Final Answer: <your complete answer>

NEVER make up tool results. ALWAYS use a tool to get real data."""

# ─────────────────────────────────────────────────────────────────────────────
# THE REACT LOOP ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_react_agent(question: str, max_iterations: int = 6) -> str:
    print(f"\nQuestion: {question}")
    print("=" * 60)

    # Conversation history — grows with each iteration
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Question: {question}"}
    ]

    for iteration in range(max_iterations):
        print(f"\n--- LOOP ITERATION #{iteration + 1} ---")

        # ── LLM CALL ──────────────────────────────────────────────────────────
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            options={
                "temperature": 0,
                "num_predict": 150,
                "repeat_penalty": 1.4,
                "stop": ["Observation:"]   # Stop before writing its own observations
            }
        )
        llm_output = response.message.content.strip()
        print(f"LLM says:\n{llm_output}")

        # Add LLM response to conversation history
        messages.append({"role": "assistant", "content": llm_output})

        # ── CHECK FOR FINAL ANSWER ─────────────────────────────────────────────
        if "Final Answer:" in llm_output:
            final = llm_output.split("Final Answer:")[-1].strip()
            print(f"\n{'='*60}")
            print(f"AGENT DONE after {iteration + 1} LLM call(s)")
            print(f"Final Answer: {final}")
            return final

        # ── PARSE ACTION (FORGIVING REGEX FOR SMALL MODELS) ────────────────────
        # Small models hallucinate formatting (e.g., "**Action:**", "Input** :")
        # We use flexible regex to catch the tool name and input no matter how messy it is.
        action_match     = re.search(r"(?i)Action[\*:\s]*(\w+)", llm_output)
        action_inp_match = re.search(r"(?i)Input[\*:\s]*[\"\']?(.*?)[\"\']?(?:\n|$)", llm_output)

        if not action_match:
            # Model didn't pick a tool — nudge it
            messages.append({"role": "user", "content": "You must specify an Action and an Action Input."})
            continue

        tool_name  = action_match.group(1).strip().lower()
        tool_input = action_inp_match.group(1).strip() if action_inp_match else ""

        # ── EXECUTE TOOL ───────────────────────────────────────────────────────
        print(f"\n>>> ACTION: {tool_name}('{tool_input}')")
        if tool_name in TOOLS:
            observation = TOOLS[tool_name](tool_input)
        else:
            observation = f"Error: Tool '{tool_name}' not found. Available: {list(TOOLS.keys())}"
        print(f">>> OBSERVATION: {observation}")

        # Feed observation back as a user message so the LLM can use it
        messages.append({"role": "user", "content": f"Observation: {observation}\nContinue."})

    return "Max iterations reached."


# ─────────────────────────────────────────────────────────────────────────────
# PART A: PLAIN LLM CALL
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART A: Regular LLM Call (No Tools, No Loop)")
print("=" * 60)
try:
    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 144 multiplied by 99? Reply with just the number."}],
        options={"num_predict": 15, "temperature": 0}
    )
    print(f"LLM answer: {r.message.content.strip()}")
    print("(This may be slightly wrong on math — LLMs sometimes hallucinate)")
except Exception as e:
    print(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PART B: AGENT WITH TOOLS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART B: Agent Call (LLM + Tools + ReAct Loop)")
print("=" * 60)

run_react_agent("What is 144 multiplied by 99? Also tell me today's date.")
