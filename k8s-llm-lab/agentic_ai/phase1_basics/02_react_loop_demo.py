"""
PHASE 1 - SCRIPT 2: Seeing the ReAct Loop in Raw Detail (Modern API)
======================================================================
In Script 1, we saw the agent run and give a final answer.
In THIS script, we make EVERY SINGLE STEP visible.

We print each message in the agent's internal conversation so you can see:
  - When the LLM decides to call a tool
  - Exactly what input it sends to the tool
  - What the tool returns
  - How the LLM uses that result to decide the next step

This is what LangSmith (LangChain's commercial tracing UI) shows you visually.
Here we replicate it manually in the terminal.

NatWest Banking Use Case:
  - FX rate lookup + calculation (multi-step)
  - Fraud flag check
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_react_agent  # LangGraph V1+
import json

# ── DEFINE TOOLS (NatWest Banking Domain) ────────────────────────────────────

@tool
def search_exchange_rate(currency_pair: str) -> str:
    """
    Look up the current exchange rate between two currencies.
    Input format: 'GBP_USD', 'EUR_GBP', 'USD_INR', 'GBP_INR'
    Returns the exchange rate as a decimal number.
    Always use this before doing any currency conversion calculation.
    """
    rates = {
        "GBP_USD": "1.27",
        "EUR_GBP": "0.84",
        "USD_INR": "83.5",
        "GBP_INR": "106.1"
    }
    rate = rates.get(currency_pair.upper())
    if rate:
        return f"1 {currency_pair[:3]} = {rate} {currency_pair[4:]}"
    return "Rate not found. Available pairs: GBP_USD, EUR_GBP, USD_INR, GBP_INR"


@tool
def calculator(expression: str) -> str:
    """
    Evaluate any mathematical expression for precise calculation.
    Use this for currency conversions, interest calculations, or any arithmetic.
    Never calculate in your head - always use this tool for accuracy.
    Input: a valid Python math expression string.
    Examples: '5000 * 1.27', '(10000 / 0.84)', 'round(6350.5, 2)'
    """
    try:
        return str(round(eval(expression), 4))
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def check_fraud_flag(transaction_id: str) -> str:
    """
    Check if a transaction has been flagged by NatWest's fraud detection system.
    Input: a transaction ID (e.g., 'TXN-8823', 'TXN-0012')
    Returns: CLEAN or FLAGGED with the reason for flagging.
    Always check this before approving high-value transactions.
    """
    flagged = {
        "TXN-8823": "FLAGGED: Unusual location - transaction attempted from Lagos, Nigeria",
        "TXN-0012": "FLAGGED: Amount exceeds daily limit of 10,000 GBP",
        "TXN-5501": "FLAGGED: Velocity alert - 15 transactions in 2 minutes"
    }
    return flagged.get(transaction_id.upper(),
                       f"{transaction_id.upper()}: CLEAN - no fraud indicators detected")


# ── SET UP THE MODERN AGENT ──────────────────────────────────────────────────
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_predict=512,
    repeat_penalty=1.3,
)
tools = [search_exchange_rate, calculator, check_fraud_flag]

agent = create_react_agent(model=llm, tools=tools)

# ── HELPER: PRINT THE AGENT TRACE STEP BY STEP ───────────────────────────────
def print_agent_trace(result: dict):
    """
    This function takes the agent's output and prints every step clearly.
    In production, this is what LangSmith / CloudWatch / Azure Monitor does
    automatically - it captures every message in the agent's conversation.
    """
    messages = result["messages"]
    step = 0

    for msg in messages:
        msg_type = type(msg).__name__

        if msg_type == "HumanMessage":
            print(f"\n{'='*60}")
            print(f"  USER QUERY")
            print(f"{'='*60}")
            print(f"  {msg.content}")

        elif msg_type == "AIMessage":
            # AI message with tool calls = LLM decided to call a tool
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                step += 1
                print(f"\n{'─'*60}")
                print(f"  LOOP STEP #{step} — LLM THINKING + ACTION")
                print(f"{'─'*60}")
                for tc in msg.tool_calls:
                    print(f"  > Decided to call tool: [{tc['name']}]")
                    print(f"  > With input: {tc['args']}")
            # AI message without tool calls = Final Answer
            elif msg.content:
                print(f"\n{'='*60}")
                print(f"  FINAL ANSWER (after {step} tool calls)")
                print(f"{'='*60}")
                print(f"  {msg.content}")

        elif msg_type == "ToolMessage":
            print(f"\n  < Tool returned: {msg.content}")
            print(f"  < LLM will now re-think with this information...")


# ── RUN THE AGENT ─────────────────────────────────────────────────────────────
# A multi-step banking query that requires 3 tool calls:
#   1. Look up GBP/USD rate
#   2. Calculate the conversion
#   3. Check fraud flag on TXN-8823
print("\n" + "="*60)
print("MULTI-STEP AGENTIC QUERY — NatWest Finance Use Case")
print("="*60)

user_query = (
    "I need two things: "
    "1) Convert 5000 GBP to USD using the current exchange rate. "
    "2) Check if transaction TXN-8823 has any fraud flags."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": user_query}]
})

print_agent_trace(result)

print("\n" + "─"*60)
print("WHAT YOU JUST SAW:")
print("  - The LLM reasoned about what tools to call (no guessing)")
print("  - Real Python functions returned real data (no hallucination)")
print("  - The agent looped through multiple steps automatically")
print("  - This EXACT pattern powers: AWS Bedrock Agents, Azure Copilot,")
print("    OpenAI Assistants, Google Vertex AI Agents")
print("─"*60)
