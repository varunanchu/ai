# Agentic AI — Classroom Concepts
**Step-by-step teaching notes from every session | Updated as we progress through each Phase**

---

## Phase 1: Foundations — What is an Agent?

---

### Lesson 1.1: Why Agents? The Problem with Regular LLM Calls

A standard LLM call is a one-shot, stateless transaction:
```
You  →  "What is 144 * 99?"
LLM  →  "It's 14255"   ← WRONG. LLMs hallucinate math.
Done.
```

**Three core weaknesses of a plain LLM call:**

| Weakness | Example | Impact |
|---|---|---|
| **Hallucination** | LLM guesses math answers instead of computing them | Wrong financial calculations |
| **No real-time data** | LLM doesn't know today's FX rate | Stale, inaccurate answers |
| **Single-shot** | Can't break a complex task into multiple steps | Fails on multi-step problems |

An **Agent** solves all three by giving the LLM *tools* it can actually call, and wrapping the whole thing in a *loop* so it can take multiple steps.

---

### Lesson 1.2: The Simplest Definition of an Agent

```
Agent = LLM Brain  +  Tools  +  A Loop
```

- **LLM Brain:** Handles the reasoning — deciding WHAT to do next
- **Tools:** Python functions that do the actual work (real math, real data)
- **Loop:** Keeps running until the LLM says it has a Final Answer

**Real-world analogy:**
Think of a junior analyst at NatWest. You give them a task: *"Find the total fraud amount for Q3 and write a summary."* They don't know the answer off the top of their head. They:
1. Open the fraud database (Tool 1)
2. Run a query (Tool 2)
3. Open Excel, do the calculation (Tool 3)
4. Write the summary

An AI Agent does exactly this — uses its "brain" to decide which tool to use at each step, and keeps going until the task is complete.

---

### Lesson 1.3: The ReAct Loop (The Engine of Every Agent)

**ReAct = Reasoning + Acting.** This is the standard pattern that powers almost every agent framework in existence — LangChain, LangGraph, CrewAI, AutoGen, Bedrock Agents all implement it.

The loop has 4 steps that repeat:

```
THOUGHT  →  The LLM reasons about what to do next
ACTION   →  The LLM picks a tool and calls it
OBSERVE  →  The Python function runs and returns a result
THOUGHT  →  The LLM reads the result and decides the next step
... repeats until ...
FINAL ANSWER  →  The LLM decides it has enough to answer
```

**Concrete Example (NatWest Finance):**
```
User: "Convert 5000 GBP to USD and check if TXN-8823 is flagged."

THOUGHT:  I need the current GBP/USD rate. I'll use the FX rate tool.
ACTION:   search_exchange_rate("GBP_USD")
OBSERVE:  "1.27"

THOUGHT:  Rate is 1.27. Now I calculate 5000 * 1.27.
ACTION:   calculator("5000 * 1.27")
OBSERVE:  "6350.0"

THOUGHT:  Good. Now I need to check the fraud flag for TXN-8823.
ACTION:   check_fraud_flag("TXN-8823")
OBSERVE:  "FLAGGED: Unusual location - transaction from Nigeria"

THOUGHT:  I now have everything. I can write the final answer.
FINAL:    "5000 GBP = 6350 USD. Transaction TXN-8823 is FLAGGED for review."
```

**Key insight:** The LLM never did any math and never guessed any data. Every number came from a real Python function. The LLM only handled the *reasoning* (deciding what to do next and how to phrase the answer).

---

### Lesson 1.4: What are Tools?

Tools are just **normal Python functions** with a `@tool` decorator. The decorator does two things:
1. Makes the function's **docstring** into a description the LLM reads to decide when to use it
2. Registers the function with the agent framework so it can be called

```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Use this tool to evaluate any mathematical expression.
    Input should be a valid Python math expression as a string.
    Example inputs: '144 * 99', '(100 + 50) / 2'
    """
    result = eval(expression)
    return str(result)
```

**The docstring is critical.** The LLM reads it to decide whether to call this tool. If your docstring is vague, the LLM won't know when to use it. Write it clearly, describing exactly what the tool does and what the input format is.

**Types of Tools in Enterprise Systems:**

| Type | What it does | Production Example |
|---|---|---|
| **Data Tools** | Query databases, call REST APIs, search Vector DBs | Pull customer records from DynamoDB |
| **Action Tools** | Make things happen in the real world | Send email, create Jira ticket, trigger pipeline |
| **Compute Tools** | Perform reliable computation | Calculate interest, run SQL, execute code |
| **Memory Tools** | Read/write to long-term memory | Save user preferences, retrieve past conversations |

---

### Lesson 1.5: Why Ollama instead of our old server.py?

| Aspect | server.py (old approach) | Ollama (new approach) |
|---|---|---|
| **Speed** | ~42 seconds per response | ~5-10 seconds per response |
| **Language** | Pure Python | Written in Go + C++ |
| **Built for** | Demo / one-shot inference | Agentic loops, local development |
| **API** | Custom FastAPI | OpenAI-compatible REST API |
| **Restart needed?** | Yes, every time | Runs as a background service |

Agents run in loops. If each loop iteration takes 42 seconds and the agent needs 4 iterations, that's **3 minutes** to answer one question. Ollama's speed brings that down to **20-40 seconds** total, which is usable for development.

---

### Lesson 1.6: How LangChain Packages the ReAct Loop

Without LangChain, implementing the ReAct loop from scratch requires:
- Writing the prompt that explains the agent format to the LLM
- Parsing the LLM output to extract the Tool name and input
- Calling the correct Python function
- Appending the result back to the prompt
- Detecting when the LLM says "Final Answer"
- Adding safety limits on max iterations

That is hundreds of lines of code. LangChain gives you the same thing in **5 lines:**

```python
from langchain.agents import create_react_agent, AgentExecutor

# 1. Pull the battle-tested ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# 2. Bundle LLM + Tools + Prompt into an agent
agent = create_react_agent(llm=llm, tools=[calculator, search_rate], prompt=prompt)

# 3. Wrap with the loop engine
agent_executor = AgentExecutor(agent=agent, tools=[...], verbose=True, max_iterations=5)

# 4. Run it
result = agent_executor.invoke({"input": "Convert 5000 GBP to USD"})
```

`AgentExecutor` is the loop engine — it repeatedly calls the LLM, parses its output, calls tools, and appends observations until it sees "Final Answer".

---

### Lesson 1.7: Observability — How Enterprise Systems Watch Agents

In production, you never run agents blind. You need to see every step for:
- **Debugging:** Why did the agent call the wrong tool?
- **Auditing:** Compliance requires a record of every AI decision
- **Cost tracking:** Each LLM call costs money — how many calls did the agent make?

LangChain provides a **Callback system** for this. Every time the agent takes a step, LangChain fires callback events. You can intercept them to log, trace, or alert.

```python
class ReActStepLogger(BaseCallbackHandler):
    def on_llm_start(self, ...):     # Fires when LLM starts thinking
    def on_tool_start(self, ...):    # Fires when a tool is called
    def on_tool_end(self, ...):      # Fires when a tool returns a result
    def on_agent_finish(self, ...):  # Fires when agent has Final Answer
```

**Production equivalent:** This is exactly what **LangSmith** (LangChain's commercial tracing tool), **AWS CloudWatch** (for Bedrock Agents), and **Azure Monitor** (for Azure ML agents) do — they implement the same callback mechanism, but store results in a searchable database with a pretty dashboard UI.

---

### Lesson 1.8: Code Walkthrough — `01_what_is_an_agent.py` (Line by Line)

**Step 1 — Connecting to Ollama:**
```python
llm = OllamaLLM(model="llama3.2", temperature=0)
```
This creates a connection to the `llama3.2` model running inside the Ollama background service.
- `model="llama3.2"` — tells it which model to use (we downloaded this, it's stored locally)
- `temperature=0` — means **zero randomness**. The model always gives its most confident answer. For agents doing math and tool lookups, you never want creative/random answers — you want reliable, deterministic reasoning.

---

**Step 2 — Defining a Tool:**
```python
@tool
def calculator(expression: str) -> str:
    """
    Use this tool to evaluate any mathematical expression.
    Input should be a valid Python math expression as a string.
    Example inputs: '144 * 99', '(100 + 50) / 2'
    """
    result = eval(expression)
    return str(result)
```
The `@tool` decorator from LangChain does two things:
1. **Registers** the function with the agent framework so it can be called during the loop
2. **Exposes the docstring** to the LLM — the LLM reads this description to decide *when* and *how* to use this tool

**Critical rule:** The docstring IS the LLM's manual for this tool. If your docstring is vague, the LLM won't know when to use it. If it's clear with examples, the LLM uses it correctly every time. This is one of the most important things to get right when building production agents.

---

**Step 3 — Pulling the ReAct Prompt from LangChain Hub:**
```python
prompt = hub.pull("hwchase17/react")
```
Writing the ReAct system prompt from scratch is complex — it needs to tell the LLM exactly how to format its Thoughts, Actions, and Observations. LangChain Hub hosts a pre-written, battle-tested version (`hwchase17/react`) that the community has refined over thousands of tests. You pull it once from the internet and it's cached locally.

This prompt essentially says to the LLM:
*"You are an agent. When answering, follow this format: Thought → Action → Observation. Keep repeating until you write 'Final Answer:'. You have these tools available: [list of tools and their descriptions]."*

---

**Step 4 — Creating the Agent:**
```python
agent = create_react_agent(llm=llm, tools=[calculator, get_current_date], prompt=prompt)
```
This bundles the three components together:
- `llm` — the brain (llama3.2 via Ollama)
- `tools` — the hands (list of Python functions it can call)
- `prompt` — the operating instructions (the ReAct format prompt)

Think of it as assembling a robot: brain + hands + instruction manual. At this point, the agent exists but isn't running yet.

---

**Step 5 — The Loop Engine (AgentExecutor):**
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=[calculator, get_current_date],
    verbose=True,        # Prints every step to the terminal
    max_iterations=5,    # Safety guard: stop if loop runs more than 5 times
    handle_parsing_errors=True
)
```
`AgentExecutor` is the **loop engine** — it's what actually *runs* the agent. It:
1. Calls the LLM with the current prompt
2. Parses the LLM output to find `Action:` and `Action Input:`
3. Calls the corresponding Python function
4. Appends the result as `Observation:` to the prompt
5. Loops back to Step 1 until it sees `Final Answer:`

**Why `max_iterations=5`?** This is a critical production safety guard. If the LLM gets confused and keeps looping (calling tools in circles), you could run up massive API costs (imagine this on GPT-4 at \$0.01 per call). Setting a max iteration limit ensures the agent gives up gracefully rather than running infinitely.

**Why `handle_parsing_errors=True`?** Smaller models like llama3.2 sometimes format their output slightly differently than the ReAct format expects (e.g., missing a colon). This flag tells the executor to recover gracefully and retry instead of crashing.

---

### Lesson 1.9: Code Walkthrough — `02_react_loop_demo.py` (The Callback System)

**What is a Callback Handler?**

In production, you never run agents blind. You need to observe every step for:
- **Debugging:** Why did the agent call the wrong tool?
- **Auditing:** Compliance requires a record of every AI decision
- **Cost tracking:** Each LLM call costs money — how many calls did the agent make?

LangChain's **Callback System** lets you intercept every agent event by subclassing `BaseCallbackHandler`:

```python
class ReActStepLogger(BaseCallbackHandler):
    def on_llm_start(self, ...):     # Fires when LLM starts a new thinking iteration
    def on_tool_start(self, ...):    # Fires just before a tool function is called
    def on_tool_end(self, ...):      # Fires when the tool returns its result
    def on_agent_finish(self, ...):  # Fires when agent reaches Final Answer
```

You attach your callback to the executor:
```python
agent_executor = AgentExecutor(agent=agent, tools=tools, callbacks=[logger])
```

**Production equivalent:** This is exactly the mechanism used by:
- **LangSmith** — stores every callback event in a searchable database with a visual timeline UI
- **AWS CloudWatch** — Bedrock Agents emit similar events to CloudWatch for monitoring
- **Azure Monitor** — Azure ML agents emit traces to Application Insights

Same pattern, different storage backends.

---

### Lesson 1.10: Why the Model Choice Matters for Agents

| Model | Size | Speed (CPU) | Tool Calling Reliability |
|---|---|---|---|
| `llama3.2:1b` | 750MB | ~3s/response | Poor — often formats output incorrectly |
| **`llama3.2` (3B)** | **2GB** | **~8s/response** | **Good — follows ReAct format reliably** |
| `llama3.2:11b` | 7GB | ~30s/response | Excellent |
| `gpt-4o` | Cloud | <1s | Perfect |

For agents, **format reliability** is more important than raw intelligence. A small model that reliably outputs `Action: calculator` and `Action Input: 144 * 99` is more useful than a large model that gives a brilliant answer but forgets to format it correctly. The llama3.2 3B model hits the right balance for local CPU development.

For production at NatWest, you would replace `OllamaLLM` with a call to your centralized Llama-3 endpoint running on vLLM/TGI on EKS — the agent code itself doesn't change at all, only the LLM connection.

---

### Lesson 1.11: The Modern LangChain/LangGraph API Migration

LangChain went through a **major architectural shift** in version 1.0+. This is one of the most important things to understand as an MLOps engineer because it affects every codebase built before 2024.

#### Old Way (LangChain 0.x) — what most tutorials still show:
```python
from langchain.agents import AgentExecutor, create_react_agent  # DEPRECATED
from langchain_ollama import OllamaLLM   # Plain text completion

llm = OllamaLLM(model="llama3.2")
agent = create_react_agent(llm=llm, tools=[...], prompt=prompt)
executor = AgentExecutor(agent=agent, tools=[...], verbose=True)
result = executor.invoke({"input": "your question"})
```

#### New Way (LangGraph 1.x) — the current enterprise standard:
```python
from langgraph.prebuilt import create_react_agent   # LangGraph takes over
from langchain_ollama import ChatOllama             # Chat model, not text completion

llm = ChatOllama(model="llama3.2")
agent = create_react_agent(model=llm, tools=[...])  # returns a compiled Graph
result = agent.invoke({"messages": [{"role": "user", "content": "your question"}]})
```

#### Why `ChatOllama` instead of `OllamaLLM`?

| | `OllamaLLM` | `ChatOllama` |
|---|---|---|
| **Type** | Text completion | Chat / instruction model |
| **Tool calling** | Via text parsing (fragile) | Via structured JSON (reliable) |
| **Output** | Plain string | `AIMessage` object |
| **Use for** | Simple text generation | Agents, tool calling, chat |

`llama3.2` natively supports **structured tool calling** — it outputs a proper JSON spec like `{"name": "calculator", "args": {"expression": "144 * 99"}}` instead of writing `Action: calculator` as text. This is 10x more reliable. `ChatOllama` enables this; `OllamaLLM` does not.

#### The MLOps Library Versioning Lesson:
This conflict happens constantly in production. Fix it with:
1. **`requirements.txt`** with pinned versions: `langchain-core==1.4.7`
2. **Virtual environments** per project
3. **`pip freeze > requirements.txt`** to snapshot a working environment

**The correct modern stack (2025):**
```
langchain >= 1.4.x
langchain-core >= 1.4.x
langchain-ollama >= 0.3.x
langgraph >= 1.2.x
```

### Lesson 1.12: CPU Inference Quirks and Defensive Coding

**The "token repeat limit reached" error:**
On CPU inference with small models (3B), Ollama sometimes enters a state where it starts predicting the same token over and over. When this happens for too long, Ollama's internal watchdog aborts with:
```
ollama._types.ResponseError: prediction aborted, token repeat limit reached
```

**Why it happens on CPU but not GPU:**
- GPU inference runs 10-20x faster, so the model processes its full context window cleanly before getting stuck
- CPU inference is slow, and the model's "attention" can drift into repetitive patterns during the slower decode steps

**The fix — three parameters that help:**
```python
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_predict=80,     # Hard limit on output length — stops runaway generation
    repeat_penalty=1.5, # Penalizes repeating tokens (1.0 = no penalty, 2.0 = very strict)
    num_ctx=1024,       # Smaller context window = less chance of drift on CPU
)
```

**The defensive coding lesson — always use try/except in agent code:**
```python
try:
    response = llm.invoke([HumanMessage(content="What is 144 * 99?")])
    print(response.content)
except Exception as e:
    print(f"LLM call failed: {e}")
    # Fallback logic here
```

In production agentic systems, LLM calls WILL fail sometimes — network timeouts, rate limits, model errors. Every LLM call in an enterprise agent should be wrapped in error handling with:
- A retry mechanism (e.g., `tenacity` library with exponential backoff)
- A fallback response ("I couldn't complete that step, please try again")
- An alert to your monitoring system (CloudWatch, Datadog)

---

### Lesson 1.13: Model Size vs Available RAM — A Critical Production Concept

**What we experienced:** `llama3.2` (3B, 2GB on disk) on an 8GB RAM machine generated complete garbage — random words, symbols, incoherent text — even for a simple "What is 5+3?" question.

**Why this happens:**
The model size on disk (2GB) is NOT the same as the memory needed to run it:
- The model loads into RAM and must ALSO leave room for:
  - The OS and background processes (~2-3GB)
  - The inference buffers (key-value cache, activation memory ~1-2x model size)
  - Python process overhead (~500MB)

```
8GB RAM Machine:
  OS + Chrome + VS Code:      ~3.5GB used
  llama3.2 3B model:          ~2.0GB
  Inference buffers:          ~1.5GB
  ─────────────────────────────────
  Total needed:               ~7.0GB
  Available headroom:         ~1.0GB  ← TOO TIGHT. Model runs unstably.
```

When the model runs without enough headroom, it starts producing nonsense tokens — like a human trying to think while starving for oxygen.

**The fix for local development on 8GB RAM:**
Always use a model that leaves at least 2-3GB of headroom:

| System RAM | Recommended Model | Size | Headroom |
|---|---|---|---|
| 8GB | `llama3.2:1b` | 1.3GB | ~3GB — safe |
| 16GB | `llama3.2` (3B) | 2.0GB | ~8GB — comfortable |
| 32GB | `llama3:8b` | 5.0GB | ~20GB — excellent |
| GPU 8GB VRAM | `llama3.2` (3B) | 2.0GB | ~5GB — great |

**Production rule of thumb:** Model size on disk × 1.5 = minimum dedicated RAM needed for stable inference.

**The debugging methodology we used:**
When an agent produces garbage or errors, always strip back to basics:
1. Test raw model output (does it generate coherent text at all?)
2. Test tool calling in isolation (does the model follow the format?)
3. Test the loop (do tools execute correctly?)
4. Only then test the full integrated system

This "peel back the layers" approach saves hours of debugging time.

### Lesson 1.14: Model Size vs Agent Capabilities (Reasoning & Formatting)

**What we experienced:** We switched to `llama3.2:1b` (1 Billion parameters) to solve our RAM constraints. It successfully answered basic chat and math questions. However, when we instructed it to format its output as `Action: calculator`, it outputted a blank string.

**The MLOps Rule of Thumb for Model Selection:**
1. **Chat/Summarization:** Small models (1B - 3B) work well. They understand language and can summarize text cleanly.
2. **Agentic Workloads (ReAct):** Require medium-to-large models (8B, 70B, or GPT-4). 

**Why do Agents need bigger models?**
Agents don't just need to know the answer; they need **formatting discipline**. The ReAct loop requires the model to output exact syntax (e.g., `Thought: ... \n Action: ...`). Small models lack the instruction-following capacity to adhere to rigid formats, causing the parsing logic in `AgentExecutor` to fail.

This is why enterprise agent orchestration (like AWS Bedrock AgentCore) typically defaults to massive frontier models like Claude 3.5 Sonnet or Anthropic Claude 3 Haiku, rather than small local models.

### Lesson 1.15: The "Ping-Pong" Flow in the Real World

To truly understand how an agent works, look at a complex real-world prompt:
> *"What is today's climate, when is the next bus, and how can I book a cab if there are no buses available?"*

**The Setup:**
- **The Brain:** The LLM
- **The Hands:** Python orchestrator (LangGraph/AgentExecutor)
- **The Tools:** `get_weather()`, `check_bus()`, `book_cab()`

**The Trace (Python ↔ LLM Ping Pong):**
1. **LLM Thought:** First, I need the weather.
   **LLM Action:** `get_weather("London")`
   **Python Observation:** "Heavy rain, 12°C"
2. **LLM Thought:** Now, check the buses.
   **LLM Action:** `check_bus("HQ")`
   **Python Observation:** "No buses running today due to strike."
3. **LLM Thought:** *(Dynamic Conditional Logic!)* The user said to book a cab IF no buses are available. I must book a cab.
   **LLM Action:** `book_cab("HQ")`
   **Python Observation:** "Success. Uber arriving in 5 mins."
4. **LLM Final Answer:** "It's heavy rain today. There are no buses, so I went ahead and booked a cab for you. It arrives in 5 minutes."

**The MLOps Takeaway:**
Before agents, developers had to hardcode complex `if/else` decision trees. With agents, the LLM dynamically creates the decision tree on the fly based on the observations returned by Python's API calls. The Python code is simply the "dumb" execution engine running the tools the LLM asks for.

---

## Phase 2: LangGraph (coming next)

*Content will be added when we start Phase 2*

---

## Phase 3: CrewAI & AutoGen (coming next)

*Content will be added when we start Phase 3*

---

## Phase 4: Model Context Protocol / MCP (coming next)

*Content will be added when we start Phase 4*

---

## Phase 5: Enterprise Orchestration — Bedrock AgentCore (coming next)

*Content will be added when we start Phase 5*

