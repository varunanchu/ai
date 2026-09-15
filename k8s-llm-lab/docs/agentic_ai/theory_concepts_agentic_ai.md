# Agentic AI — Theory Concepts
**From Basic to Enterprise Level | NatWest GenAI Platform Engineering**

---

## 1. What is an AI Agent?

A standard LLM call is a one-shot transaction:
> User sends a prompt → LLM generates a response → Done.

An **AI Agent** is fundamentally different. It is a *loop*.
> User gives a goal → Agent *decides what to do* → Agent *uses tools* → Agent *observes the result* → Agent *decides the next step* → Repeats until the goal is achieved.

**The simplest definition:**
```
Agent = LLM Brain + Tools + A Loop (Think → Act → Observe → Think again)
```

**Real-world analogy:** A junior analyst at NatWest. You give them a task: "Find me the total fraud amount for Q3 and write a summary." They don't know the answer. They search the database (Tool 1), open Excel (Tool 2), calculate (Tool 3), and write the summary. They use their brain at every step to decide what to do next.

---

## 2. The ReAct Loop (The Engine of Every Agent)

ReAct stands for **Re**asoning + **Act**ing. It is the standard pattern that powers almost all agents (LangChain, LangGraph, CrewAI, AutoGen all implement this).

```
Step 1 → THOUGHT:   "The user wants to know today's date and the square root of 144."
Step 2 → ACTION:    Call tool: get_current_date()
Step 3 → OBSERVE:   Tool returned: "15 September 2026"
Step 4 → THOUGHT:   "I have the date. Now I need to calculate sqrt(144)."
Step 5 → ACTION:    Call tool: calculator(expression="sqrt(144)")
Step 6 → OBSERVE:   Tool returned: "12"
Step 7 → THOUGHT:   "I have both answers. I can now form the final response."
Step 8 → FINAL:     "Today is 15 September 2026. The square root of 144 is 12."
```

This loop runs entirely inside the LLM. The "thought" steps happen in the LLM's reasoning, and the "action" steps are function calls your code handles.

---

## 3. What are Tools?

Tools are just **Python functions** that an agent is allowed to call. You decorate them with a special annotation so the LLM knows they exist. The LLM decides **when** and **how** to call them based on its reasoning.

```python
# Any regular Python function can become a Tool
def search_hr_policy(query: str) -> str:
    """Search the NatWest HR policy database for an answer."""
    # In production, this queries a Vector DB or internal API
    return "Maternity leave at NatWest is 26 weeks at full pay."
```

Tool types in enterprise systems:
- **Data Tools:** Query a database, call a REST API, search a Vector DB (RAG)
- **Action Tools:** Send an email, create a Jira ticket, trigger a pipeline
- **Compute Tools:** Run a calculation, execute code, run a SQL query
- **Memory Tools:** Write to/read from long-term memory stores

---

## 4. Memory in Agents

| Memory Type | What It Is | Example | Storage |
|---|---|---|---|
| **In-Context (Short-term)** | The conversation history in the current prompt | "You said my name is Varun" | LLM's context window |
| **External (Long-term)** | Facts stored outside the LLM between sessions | User preferences, past decisions | Vector DB, Redis |
| **Entity Memory** | Facts about specific things/people | "Varun works at NatWest" | Structured JSON/DB |
| **Episodic Memory** | Summarized history of past conversations | "Last week we discussed LoRA" | Summarization + Storage |

---

## 5. Framework Comparison (When to Use Which)

| Framework | Metaphor | Best For | Complexity |
|---|---|---|---|
| **LangChain** | A toolkit / Swiss Army knife | Building chains, RAG pipelines, single agents with tools | Low |
| **LangGraph** | A state machine / flowchart | Multi-step, stateful agents with conditional routing | Medium |
| **CrewAI** | A corporate team structure | Multiple agents with defined Roles + Tasks + Process | Medium |
| **AutoGen** | A roundtable meeting | Multiple agents talking to each other to solve a problem | Medium-High |
| **MCP** | A USB standard for AI | Standardizing how ANY agent connects to ANY tool | Protocol level |
| **Bedrock AgentCore** | A managed enterprise platform | Running agents in production on AWS without managing infra | Cloud level |

---

## 6. LangChain (The Foundation)

LangChain is the foundational toolkit. You rarely use it alone in production, but every other framework (LangGraph, CrewAI) is built on top of it.

Key building blocks:
- **PromptTemplate:** A structured, reusable prompt with variables.
- **Chain:** Connecting a prompt → LLM → output parser in a pipeline.
- **Runnable:** The modern interface for composing chains (`.pipe()` syntax).
- **AgentExecutor:** Wraps an LLM and a list of Tools, and runs the ReAct loop for you.

---

## 7. LangGraph (The Orchestrator)

LangGraph solves one key problem with raw agents: **They can loop forever.**
If you give an agent a task and it gets confused, it can keep calling tools in a loop and never stop.

LangGraph enforces structure. You define a **Graph** with:
- **Nodes:** Functions that do something (e.g., "call HR agent", "call Finance agent")
- **Edges:** Connections between nodes (e.g., "after routing, go to the right specialist")
- **Conditional Edges:** Branching logic (e.g., "if topic = HR, go to HR node. Else go to IT node.")
- **State:** A shared dictionary that every node can read from and write to

This makes agent behavior **deterministic, auditable, and production-safe**.

---

## 8. CrewAI (Multi-Agent Teams)

CrewAI models a company org chart. Key concepts:
- **Agent:** Has a `role`, a `goal`, and a `backstory` (its system prompt).
- **Task:** A specific piece of work given to an Agent, with an `expected_output`.
- **Crew:** A collection of Agents + Tasks.
- **Process:** How tasks are done — `sequential` (one by one) or `hierarchical` (a manager agent oversees workers).

---

## 9. AutoGen (Conversational Multi-Agent)

AutoGen (by Microsoft) is different — agents actually *talk to each other* to collaboratively solve a problem. Key concepts:
- **AssistantAgent:** The LLM-powered agent that reasons and generates responses.
- **UserProxyAgent:** An agent that can execute code or act as a human-in-the-loop.
- **GroupChat:** Multiple agents in a shared conversation, with a GroupChatManager deciding who speaks next.

---

## 10. Model Context Protocol (MCP)

MCP is an **open standard** created by Anthropic to solve tool fragmentation.

**The Problem Before MCP:**
Every agent framework (LangChain, AutoGen, CrewAI) had its own way of defining tools. If you built a NatWest HR policy tool for LangChain, you had to rewrite it from scratch for CrewAI, and again for AutoGen. 100 tools × 5 frameworks = 500 implementations.

**The Solution (MCP):**
MCP is like a USB standard. You write the tool once as an **MCP Server**, and any **MCP Client** (LangGraph, Claude Desktop, AutoGen, any future framework) can connect to it and use it automatically.

**Architecture:**
```
MCP Client (Agent)  ←──── MCP Protocol ────→  MCP Server (Your Tools)
(LangGraph, Claude)           (JSON-RPC)        (HR Search, DB Query, etc.)
```

---

## 11. AWS Bedrock AgentCore

Bedrock AgentCore is AWS's fully managed multi-agent platform. Instead of managing EKS clusters for your agents, AWS runs and scales them for you.

Key components:
- **Agent:** The LLM + instructions + tools. Configured via AWS Console or Terraform.
- **Action Groups:** The tools the agent can call (Lambda functions or OpenAPI specs).
- **Knowledge Base:** A managed RAG store (Bedrock embeds documents and stores in OpenSearch automatically).
- **Collaborator Agents:** Multiple specialist agents that a supervisor agent can delegate to (native Multi-Agent support).
- **Memory:** Built-in session memory managed by AWS.

---

## 12. Multi-Agent Orchestration Patterns

| Pattern | How It Works | Best For | Example |
|---|---|---|---|
| **Sequential** | Agent A finishes → passes result to Agent B | Simple linear workflows | Write report → Review report |
| **Supervisor** | A manager agent delegates tasks to specialist agents | Complex workflows with routing | Router → HR/Finance/IT specialist |
| **Hierarchical** | Manager → Sub-manager → Workers | Very large enterprise workflows | CTO → Manager → Engineers |
| **Peer-to-Peer** | Agents talk directly to each other | Collaborative problem solving | Debate between 2 agents |
| **Swarm** | Dynamic delegation, any agent can hand off to any other | Unpredictable, open-ended tasks | OpenAI Swarm |

