# AWS Bedrock Agents vs. Phase 1 Local Agents

When moving from a local script (like our `01_what_is_an_agent.py`) to an enterprise cloud environment like **AWS Bedrock**, the core concepts remain exactly the same, but the architecture shifts to managed cloud services.

Here is the exact mapping of what we just learned to AWS Bedrock components:

## 1. The Components Mapping

| Phase 1 Local Concept | AWS Bedrock Equivalent | Explanation |
| :--- | :--- | :--- |
| **The LLM** (`llama3.2`) | **Foundation Model** (e.g., Claude 3.5 Sonnet) | You select a foundational model from the Bedrock console to act as the "Brain". |
| **The ReAct Loop** (Our custom Python `while` loop) | **Bedrock Managed Orchestration** | AWS completely hides the ReAct loop from you. You do not write the parsing logic. AWS handles the `Thought -> Action -> Observation` ping-pong natively on their servers. |
| **Tools** (Python functions like `calculator()`) | **Action Groups** (AWS Lambda + OpenAPI Schema) | In AWS, a tool consists of two parts: <br>1. An **OpenAPI JSON/YAML schema** that describes what the tool does (this replaces our `SYSTEM_PROMPT` descriptions).<br>2. An **AWS Lambda function** that executes the code. |
| **Memory** (Appending to `messages` array) | **Session State** (`sessionId`) | When you call Bedrock, you pass a `sessionId`. AWS automatically stores the conversation history in its managed backend. You don't need to append messages manually. |

---

## 2. How the Flow Works in AWS Bedrock

1. **Trigger:** Your frontend (e.g., a React app or Streamlit) calls a Python backend.
2. **Boto3 Invocation:** Your Python backend calls `boto3.client('bedrock-agent-runtime').invoke_agent(...)` passing the user's prompt.
3. **AWS Internal ReAct:** 
   - AWS Bedrock sends the prompt to Claude.
   - Claude replies with a JSON tool call (e.g., "I need to use the calculate API").
   - AWS Bedrock pauses Claude, triggers your **AWS Lambda function**, gets the result, and feeds it *back* to Claude.
4. **Response:** Bedrock streams the final answer back to your `boto3` client.

*(See `agentic_ai/phase1_basics/03_aws_bedrock_boto3_example.py` for the exact code).*

---

## 3. Are we going to use MCP (Model Context Protocol) at this stage?

**Short answer: No, not natively with AWS Bedrock.**

**What is MCP?**
Model Context Protocol (MCP) is a new open standard created by Anthropic. It allows an LLM to easily connect to local data sources or tools (like reading a local SQLite database, or your local Github repo) without writing custom API wrappers. 

**Why we aren't using it for Bedrock yet:**
1. AWS Bedrock Agents primarily use **OpenAPI Schemas + Lambda** as their standard for defining tools. 
2. MCP is brilliant for *local* agents (like building an agent on your laptop to read your local files) or connecting Claude Desktop to your local environment. 
3. While you *can* build an MCP server and expose it via AWS API Gateway to Bedrock, it adds unnecessary complexity for basic cloud tool calling.

**Our Roadmap:**
- We will learn **MCP** specifically in **Phase 4**, where we will build a local MCP server to connect an agent to a local database securely.
- For cloud orchestration (Phase 5), we will stick to the standard Bedrock Action Group (Lambda/OpenAPI) architecture, as that is what NatWest and other enterprises mandate for security and infrastructure-as-code (Terraform) reasons.

