"""
PHASE 2 - SCRIPT 3: Memory and "Loops" in LangGraph
===================================================
Real chatbots don't run in one single execution. They pause, wait for 
the user to reply, and then resume. 

To do this, LangGraph uses Checkpointers (MemorySaver) to save the 
State across multiple runs using a `thread_id`.

USE CASE: NatWest Balance Checker
- If the user doesn't provide an account number, the bot asks for it.
- The user replies with the number.
- The bot REMEMBERS the previous context and fetches the balance.
"""

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import re

# ── 1. DEFINE THE STATE (WITH MEMORY) ───────────────────────────────────────
# We use `Annotated[list, operator.add]` to tell LangGraph to APPEND to the 
# list, rather than overwrite it, creating a true conversation history!
class ChatState(TypedDict):
    current_input: str
    account_number: str
    history: Annotated[list[str], operator.add] 
    bot_response: str


# ── 2. DEFINE THE NODES ──────────────────────────────────────────────────────

def extract_account_node(state: ChatState):
    """NODE 1: Tries to find a 6-digit account number in the input."""
    print("  [System] Running Extraction Node...")
    
    # We use basic Python regex to find a 6-digit number
    # (In a larger system, you'd use a small LLM or NLP tool here)
    match = re.search(r'\b\d{6}\b', state["current_input"])
    
    account_num = match.group(0) if match else ""
    
    # Notice we return ONLY the fields we want to update
    return {
        "account_number": account_num,
        "history": [f"User: {state['current_input']}"] # Append user msg to history
    }

def ask_user_node(state: ChatState):
    """NODE 2A: Triggered if account number is missing."""
    print("  [System] Running 'Ask User' Node...")
    response = "I can help with your balance. Please provide your 6-digit account number."
    
    return {
        "bot_response": response,
        "history": [f"Bot: {response}"] # Append bot msg to history
    }

def fetch_balance_node(state: ChatState):
    """NODE 2B: Triggered if account number IS present."""
    print("  [System] Running 'Fetch Balance' Node...")
    # Mocking a database call
    response = f"Thank you. The current balance for account {state['account_number']} is £5,420.00."
    
    return {
        "bot_response": response,
        "history": [f"Bot: {response}"] # Append bot msg to history
    }


# ── 3. DEFINE THE ROUTER (CONDITIONAL EDGE) ──────────────────────────────────

def router(state: ChatState):
    """Decides where to go based on the state memory."""
    # If the state already has an account number, fetch the balance!
    if state.get("account_number"):
        return "fetch_balance"
    else:
        return "ask_user"


# ── 4. BUILD THE GRAPH WITH MEMORY ───────────────────────────────────────────

builder = StateGraph(ChatState)

builder.add_node("extract", extract_account_node)
builder.add_node("ask_user", ask_user_node)
builder.add_node("fetch_balance", fetch_balance_node)

builder.add_edge(START, "extract")

# The condition that creates our workflow logic
builder.add_conditional_edges(
    "extract", 
    router, 
    {
        "fetch_balance": "fetch_balance",
        "ask_user": "ask_user"
    }
)

builder.add_edge("ask_user", END)
builder.add_edge("fetch_balance", END)

# === THE MAGIC HAPPENS HERE ===
# We add a MemorySaver. This saves the State to RAM so when the 
# user replies 5 minutes later, the bot remembers everything.
memory = MemorySaver()
agent = builder.compile(checkpointer=memory)


# ── 5. RUN THE INTERACTIVE SIMULATION ────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING NATWEST STATEFUL CHATBOT")
    print("="*60)
    
    # We define a thread_id. This represents a specific user's conversation session.
    config = {"configurable": {"thread_id": "varun_session_001"}}
    
    # --- TURN 1 ---
    print("\n[TURN 1: User asks a generic question]")
    input_1 = "What is the balance of my account?"
    print(f"Customer: {input_1}")
    
    # Invoke the graph with the config!
    state_1 = agent.invoke({"current_input": input_1}, config=config)
    print(f"Bot: {state_1['bot_response']}")
    
    
    # --- TURN 2 ---
    print("\n[TURN 2: User provides the missing information]")
    # Notice the user doesn't say "balance" again. The bot has to REMEMBER that!
    input_2 = "My number is 123456."
    print(f"Customer: {input_2}")
    
    # We invoke it again WITH THE SAME THREAD_ID. 
    # LangGraph automatically loads the old history and the fact that we were looking for a balance!
    state_2 = agent.invoke({"current_input": input_2}, config=config)
    print(f"Bot: {state_2['bot_response']}")
    
    # --- PROOF OF MEMORY ---
    print("\n" + "="*60)
    print("PROOF OF STATE PERSISTENCE (CONVERSATION HISTORY):")
    print("="*60)
    for msg in state_2["history"]:
        print(msg)

