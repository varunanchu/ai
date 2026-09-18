"""
LANGCHAIN DEEP DIVE: The Core Building Blocks
=============================================================
Before building complex agents, you must understand LCEL 
(LangChain Expression Language). LCEL uses the pipe `|` 
operator to chain components together like a factory assembly line.

In this script, we will cover the 3 most common LangChain pipelines:
1. The Basic Chain (Prompt -> Model -> String)
2. The Chat Persona Chain (System Message + User Message)
3. The Data Extraction Chain (Forcing JSON Output)
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize our tiny local model
llm = ChatOllama(model="llama3.2:1b", temperature=0, num_predict=150)

print("\n" + "="*60)
print("FEATURE 1: THE BASIC LCEL CHAIN")
print("="*60)
# -------------------------------------------------------------------------
# Feature 1 Explanation:
# Instead of manually inserting strings, we use PromptTemplate. 
# StrOutputParser() automatically strips away weird metadata from the LLM 
# and just gives you the clean string text.
# 
# Pipeline: Input Dict -> Prompt -> LLM -> Clean String
# -------------------------------------------------------------------------

basic_prompt = PromptTemplate.from_template("Translate the following word to French: {word}")

# THIS is LCEL. It reads: "Take the prompt, pipe it to the LLM, pipe it to the Parser"
basic_chain = basic_prompt | llm | StrOutputParser()

# We trigger the chain using .invoke() and pass the dictionary variable
result_1 = basic_chain.invoke({"word": "Hello"})
print(f"Input: 'Hello' -> Output: {result_1.strip()}")


print("\n" + "="*60)
print("FEATURE 2: THE CHAT PERSONA CHAIN")
print("="*60)
# -------------------------------------------------------------------------
# Feature 2 Explanation:
# In the real world, you rarely use basic prompts. You use ChatPrompts.
# A ChatPrompt separates instructions (System) from the user's input (Human).
# This prevents users from "jailbreaking" your agent.
# -------------------------------------------------------------------------

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grumpy, sarcastic IT support agent. Give very short, unhelpful answers."),
    ("human", "My computer won't turn on, what should I do {name}?")
])

chat_chain = chat_prompt | llm | StrOutputParser()

result_2 = chat_chain.invoke({"name": "Varun"})
print("User: My computer won't turn on, what should I do Varun?")
print(f"Grumpy IT Agent: {result_2.strip()}")


print("\n" + "="*60)
print("FEATURE 3: STRUCTURED DATA EXTRACTION (JSON)")
print("="*60)
# -------------------------------------------------------------------------
# Feature 3 Explanation:
# LLMs naturally output conversational text ("Here is the data you requested...").
# In MLOps, we hate conversational text. We want JSON so our Python code 
# can read variables directly (e.g., data['account_number']).
# JsonOutputParser forces the LLM to output a valid Python dictionary.
# -------------------------------------------------------------------------

# We define the parser
json_parser = JsonOutputParser()

# We pass the parser's formatting instructions directly into the prompt!
json_prompt = PromptTemplate(
    template="""Extract the customer's name and account balance from this email.
    
    Email: {email_text}
    
    {format_instructions}""",
    input_variables=["email_text"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()}
)

# Pipeline: Input -> Prompt -> LLM -> JSON Parser (Converts string to Python Dict)
json_chain = json_prompt | llm | json_parser

email = "Hi NatWest, this is John Doe. I am looking at my statement and I see my balance is 5400 GBP. Can you verify?"

print(f"Reading Email: '{email}'...\n")

try:
    result_3 = json_chain.invoke({"email_text": email})
    
    # Notice how result_3 is NOT a string. It is a real Python dictionary!
    print("Python Dictionary Output:")
    print(f"Type: {type(result_3)}")
    print(f"Extracted Name: {result_3.get('name', 'Not found')}")
    print(f"Extracted Balance: {result_3.get('balance', 'Not found')}")
    
except Exception as e:
    # Small 1B models sometimes fail to write perfect JSON brackets. 
    # In production with Llama-3 8B or Claude, this rarely fails.
    print(f"[Note] The 1B model failed to output perfect JSON. Error: {e}")

print("\n" + "="*60)
print("LANGCHAIN DEEP DIVE COMPLETE")
print("="*60)

