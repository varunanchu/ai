"""
PHASE 1 to AWS BEDROCK MAPPING: How this works in the Enterprise
================================================================

In our local Phase 1, we wrote the ReAct loop and the Python tools manually. 
In AWS Bedrock, Amazon manages the loop for you. You just configure the pieces 
and call the Agent using the `boto3` SDK.

Here is what the Python code looks like when calling a fully built AWS Bedrock Agent.
"""

import boto3
import uuid

def call_bedrock_agent(prompt: str, agent_id: str, agent_alias_id: str):
    """
    This is how you invoke an Agent in AWS. 
    Notice there is NO ReAct loop here. Bedrock handles the loop internally!
    """
    # Create the Bedrock runtime client
    client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

    # A unique ID for this user's conversation (This handles MEMORY automatically!)
    session_id = str(uuid.uuid4())

    print(f"Sending prompt to Bedrock Agent: {prompt}")
    
    # ── THE MAGIC HAPPENS HERE ────────────────────────────────────────────────
    # When we call invoke_agent, AWS spins up Claude, feeds it the prompt, 
    # executes the ReAct loop internally, calls your Lambda functions (tools), 
    # and returns only the final answer.
    response = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        sessionId=session_id,
        inputText=prompt
    )
    
    # Process the streaming response from Bedrock
    completion = ""
    for event in response.get('completion'):
        chunk = event.get('chunk')
        if chunk:
            completion += chunk.get('bytes').decode()
            
    return completion


# ==============================================================================
# HOW TOOLS WORK IN AWS BEDROCK (Action Groups)
# ==============================================================================
# In our local script, a tool was just a Python function:
# def calculator(expression): ...
#
# In AWS Bedrock, a tool is an AWS Lambda Function paired with an OpenAPI schema.
# Here is what the Lambda function (Tool) looks like:

def lambda_handler(event, context):
    """
    AWS Lambda acting as a Bedrock Tool (Action Group)
    """
    # 1. Bedrock tells the Lambda which tool the LLM wants to use
    api_path = event['apiPath']
    
    # 2. Extract the arguments the LLM provided (e.g., the math expression)
    parameters = event.get('parameters', [])
    
    result = ""
    
    if api_path == '/calculate':
        expression = next((p['value'] for p in parameters if p['name'] == 'expression'), "")
        try:
            result = str(eval(expression))
        except Exception as e:
            result = f"Error: {e}"
            
    # 3. Return the observation back to Bedrock's internal ReAct loop
    response_body = {
        'application/json': {
            'body': result
        }
    }
    
    return {
        'messageVersion': '1.0',
        'response': {
            'actionGroup': event['actionGroup'],
            'apiPath': event['apiPath'],
            'httpMethod': event['httpMethod'],
            'httpStatusCode': 200,
            'responseBody': response_body
        }
    }

if __name__ == "__main__":
    print("This is a conceptual mapping file for AWS Bedrock.")
    print("To run this, you would need actual AWS Agent IDs and credentials.")

