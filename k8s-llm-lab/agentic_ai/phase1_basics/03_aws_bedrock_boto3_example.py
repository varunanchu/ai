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

# ==============================================================================
# PART 1: BUILD-TIME (Control Plane) - Where Models & Tools are Configured
# ==============================================================================
# In AWS, you don't define the model and tools in your runtime EKS/ECS code.
# You define them when you CREATE the Agent (via Terraform, Console, or Boto3 Control Plane).
#
# Here is how an MLOps engineer configures the Agent in AWS:

def pseudo_create_agent_infrastructure():
    # We use 'bedrock-agent' (Control Plane), not runtime!
    client = boto3.client('bedrock-agent')

    # 1. CONFIGURE THE MODEL AND PROMPT
    response = client.create_agent(
        agentName='NatWest-Support-Agent',
        foundationModel='anthropic.claude-3-sonnet-20240229-v1:0', # <--- MODEL DEFINED HERE
        instruction='You are a helpful assistant. Use tools to solve math.'
    )
    agent_id = response['agent']['agentId']

    # 2. CONFIGURE THE TOOLS (ACTION GROUP)
    # You link the Agent to a Lambda function and an OpenAPI schema stored in S3.
    client.create_agent_action_group(
        agentId=agent_id,
        actionGroupName='Calculator-Tools',
        actionGroupExecutor={
            'lambda': 'arn:aws:lambda:us-east-1:123456789012:function:my-calculator-lambda' # <--- TOOL DEFINED HERE
        },
        apiSchema={
            's3': {
                's3BucketName': 'natwest-openapi-schemas',
                's3ObjectKey': 'calculator-schema.json' # Tells the LLM exactly what the Lambda does
            }
        }
    )
    
    # AWS prepares it, and outputs an Alias ID used for runtime.
    print(f"Agent built! Give this Agent ID to the EKS developers: {agent_id}")


# ==============================================================================
# PART 2: RUN-TIME (Data Plane) - What runs on your EKS/ECS Cluster
# ==============================================================================
# This is the code that actually receives the request from your Frontend UI.
# It doesn't know about Claude or Lambda - it just calls the Agent ID.

def call_bedrock_agent(prompt: str, agent_id: str, agent_alias_id: str):
    """
    This runs on your EKS container.
    """
    # Create the Bedrock RUNTIME client
    client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

    # A unique ID for this user's conversation (This handles MEMORY automatically!)
    session_id = str(uuid.uuid4())

    print(f"Sending prompt to Bedrock Agent: {prompt}")
    
    # ── THE MAGIC HAPPENS HERE ────────────────────────────────────────────────
    # AWS spins up Claude, feeds it the prompt, executes the ReAct loop internally, 
    # calls the Lambda functions configured in Part 1, and returns the final answer.
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
# PART 3: THE LAMBDA FUNCTION (The actual Tool execution)
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

