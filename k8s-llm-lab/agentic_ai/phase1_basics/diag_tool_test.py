"""
Diagnostic: Test if llama3.2 can do native tool calling via raw Ollama Python API.
This bypasses LangChain completely to isolate the issue.
"""
import ollama

# Simple direct tool calling test using the raw Ollama client
response = ollama.chat(
    model='llama3.2',
    messages=[{
        'role': 'user',
        'content': 'What is 144 multiplied by 99? Use the calculator tool.'
    }],
    tools=[{
        'type': 'function',
        'function': {
            'name': 'calculator',
            'description': 'Evaluate a mathematical expression. Input: expression string like "144 * 99"',
            'parameters': {
                'type': 'object',
                'properties': {
                    'expression': {
                        'type': 'string',
                        'description': 'A valid math expression'
                    }
                },
                'required': ['expression']
            }
        }
    }]
)

print("Model response:")
print(f"  Content: {response.message.content}")
print(f"  Tool calls: {response.message.tool_calls}")
print(f"  Stop reason: {response.done_reason}")

