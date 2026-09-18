"""
PHASE 2 - SCRIPT 1: The LangGraph Router Pattern
=================================================
Instead of a fragile 'while' loop, we build a robust State Machine.
This is the NatWest Support Desk Router. 

FLOWCHART:
[START] -> [Classify Node] ---> if 'fraud' ---> [Fraud Node] -----> [END]
                          |---> if 'loan'  ---> [Loan Node] ------> [END]
                          |---> else       ---> [General Node] ---> [END]
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ── 1. DEFINE THE STATE ───────────────────────────────────────────────────────
# The State is the memory object that gets passed to every Node in the graph.
# It holds the user's message, the classification result, and the final answer.
class AgentState(TypedDict):
    user_query: str
    category: str
    final_response: str


# ── 2. DEFINE THE NODES (The blocks in our flowchart) ──────────────────────────

def classify_node(state: AgentState):
    """Node 1: Uses the LLM to classify the user's query."""
    print("--- [NODE] CLASSIFY TICKET ---")
    
    # We use 'Few-Shot Prompting' to show the 1B model EXACTLY what to output.
    prompt = f"""Classify the user query into exactly one category: FRAUD, LOAN, or GENERAL.

Example 1:
Query: Help, my card was stolen and there are unauthorized charges!
Category: FRAUD

Example 2:
Query: What are the current interest rates for a mortgage?
Category: LOAN

Example 3:
Query: How do I change my address?
Category: GENERAL

Now classify this query. Output ONLY the category word.
Query: {state['user_query']}
Category:"""

    llm = ChatOllama(model="llama3.2:1b", temperature=0, num_predict=10)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_output = response.content.strip().lower()
    
    print(f"  [Debug] Raw LLM Output: '{raw_output}'")
    
    # Clean up the output
    if "fraud" in raw_output: category = "fraud"
    elif "loan" in raw_output: category = "loan"
    else: category = "general"
    
    print(f"  -> LLM classified as: {category.upper()}")
    
    # Update the state with the category
    return {"category": category}


def fraud_department_node(state: AgentState):
    """Node 2A: Handles fraud tickets"""
    print("--- [NODE] FRAUD DEPARTMENT ---")
    response = "FRAUD ALERT TRIGGERED: Your card has been temporarily frozen. A fraud specialist will call you in 5 minutes at your registered number."
    return {"final_response": response}

def loan_department_node(state: AgentState):
    """Node 2B: Handles loan tickets"""
    print("--- [NODE] LOAN DEPARTMENT ---")
    response = "MORTGAGE TEAM: Current NatWest base rate is 5.25%. Please visit the nearest branch to speak with a mortgage advisor."
    return {"final_response": response}

def general_department_node(state: AgentState):
    """Node 2C: Handles everything else"""
    print("--- [NODE] GENERAL SUPPORT ---")
    
    # We could call an LLM here, but we'll keep it simple for the demo
    response = "GENERAL SUPPORT: Thank you for contacting NatWest. How can I assist you with your account today?"
    return {"final_response": response}


# ── 3. DEFINE THE CONDITIONAL ROUTER (The arrows in our flowchart) ───────────

def route_ticket(state: AgentState):
    """
    This is standard Python code. It looks at the state and decides 
    which Node to go to next. No LLM hallucinations allowed here!
    """
    category = state.get("category", "general")
    print(f"--- [ROUTER] Routing ticket based on category: {category} ---")
    
    if category == "fraud":
        return "fraud_node"
    elif category == "loan":
        return "loan_node"
    else:
        return "general_node"


# ── 4. BUILD THE GRAPH ───────────────────────────────────────────────────────

# Initialize the graph with our State object
builder = StateGraph(AgentState)

# Add all our nodes to the graph
builder.add_node("classifier", classify_node)
builder.add_node("fraud_node", fraud_department_node)
builder.add_node("loan_node", loan_department_node)
builder.add_node("general_node", general_department_node)

# Define the flow (Edges)
builder.add_edge(START, "classifier") # Always start by classifying

# Add the conditional edges (if this, go there)
builder.add_conditional_edges(
    "classifier", # The node we are leaving
    route_ticket, # The Python function that decides where to go
    {
        "fraud_node": "fraud_node",
        "loan_node": "loan_node",
        "general_node": "general_node"
    }
)

# All departments go to the END when finished
builder.add_edge("fraud_node", END)
builder.add_edge("loan_node", END)
builder.add_edge("general_node", END)

# Compile the graph into an executable agent!
agent = builder.compile()


# ── 5. RUN THE TESTS ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "Help! I just saw a charge for $500 in another country that I didn't make!",
        "What are the current interest rates for a 30-year fixed mortgage?",
        "How do I change the address on my current account?"
    ]

    print("\n" + "="*60)
    print("STARTING NATWEST LANGGRAPH ROUTER")
    print("="*60)

    for i, query in enumerate(test_queries):
        print(f"\n\nTEST {i+1}: '{query}'")
        print("-" * 40)
        
        # Invoke the graph with the initial state
        initial_state = {"user_query": query}
        
        # The graph will run all the nodes and return the final state
        final_state = agent.invoke(initial_state)
        
        print(f"\nFINAL OUTPUT TO CUSTOMER:\n{final_state['final_response']}")

