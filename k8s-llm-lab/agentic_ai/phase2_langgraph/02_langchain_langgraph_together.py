"""
PHASE 2 - SCRIPT 2: LangChain + LangGraph Working Together
===========================================================
This script demonstrates how LangChain (the building blocks) 
lives INSIDE LangGraph (the orchestrator).

USE CASE: A Financial Report Generator.
1. Data Node fetches raw text.
2. Summarize Node uses a LangChain LCEL pipeline to summarize.
3. Format Node uses another LangChain pipeline to create the final report.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# --- LANGCHAIN IMPORTS (The Building Blocks) ---
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── 1. DEFINE THE STATE (LANGGRAPH) ──────────────────────────────────────────
class ResearchState(TypedDict):
    topic: str
    raw_data: str
    summary: str
    final_report: str

# ── 2. DEFINE THE NODES (USING LANGCHAIN INSIDE THEM) ────────────────────────

def fetch_data_node(state: ResearchState):
    """NODE 1: Simulates fetching raw data from an API."""
    print(f"\n--- [NODE 1: FETCH DATA] Searching for {state['topic']} ---")
    
    # Mock data for demonstration
    mock_data = f"{state['topic']} reported a revenue increase of 15% to $90 billion. Net income was $20 billion. The CEO announced a new AI product launch."
    
    return {"raw_data": mock_data}


def summarize_node(state: ResearchState):
    """
    NODE 2: Uses LangChain (LCEL) to summarize the data.
    This shows LangChain living perfectly inside a LangGraph node!
    """
    print("\n--- [NODE 2: SUMMARIZE] Using LangChain to summarize ---")
    
    # 1. LangChain Prompt
    prompt = PromptTemplate.from_template(
        "Extract the key financial numbers from this text. Keep it very short: {text}"
    )
    
    # 2. LangChain Model
    llm = ChatOllama(model="llama3.2:1b", temperature=0, num_predict=50)
    
    # 3. LangChain Parser
    parser = StrOutputParser()
    
    # 4. LangChain LCEL Pipeline (Prompt -> Model -> Parser)
    # This is pure LangChain code!
    chain = prompt | llm | parser
    
    # Execute the LangChain pipeline using the state data
    summary_result = chain.invoke({"text": state["raw_data"]})
    print(f"  [Debug] LangChain Summary output: {summary_result.strip()}")
    
    return {"summary": summary_result}


def format_report_node(state: ResearchState):
    """
    NODE 3: Uses another LangChain pipeline to format the final output.
    """
    print("\n--- [NODE 3: FORMAT] Using LangChain to create markdown ---")
    
    prompt = PromptTemplate.from_template(
        "Write a 2-sentence professional executive summary using this data:\n{summary}"
    )
    llm = ChatOllama(model="llama3.2:1b", temperature=0, num_predict=50)
    chain = prompt | llm | StrOutputParser()
    
    final_report = chain.invoke({"summary": state["summary"]})
    
    return {"final_report": final_report}


# ── 3. BUILD THE GRAPH (LANGGRAPH ORCHESTRATION) ─────────────────────────────
# We use LangGraph to map out how the nodes talk to each other.

builder = StateGraph(ResearchState)

builder.add_node("fetch", fetch_data_node)
builder.add_node("summarize", summarize_node)
builder.add_node("format", format_report_node)

# Flow: Start -> Fetch -> Summarize -> Format -> End
builder.add_edge(START, "fetch")
builder.add_edge("fetch", "summarize")
builder.add_edge("summarize", "format")
builder.add_edge("format", END)

# Compile into a working application
agent = builder.compile()


# ── 4. RUN IT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING FINANCIAL REPORT WORKFLOW (LangChain + LangGraph)")
    print("="*60)
    
    initial_state = {"topic": "NatWest Q3 2024 Earnings"}
    final_state = agent.invoke(initial_state)
    
    print("\n" + "="*60)
    print("FINAL EXECUTIVE REPORT:")
    print("="*60)
    print(final_state["final_report"].strip())

