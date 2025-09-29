import requests

prompt = """
You are a senior software engineer specializing in code quality, refactoring, and design patterns.

I will provide you with a code snippet that contains the "Ignoring monitoring for data drift" misuse.

Here is a description of the misuse:
it refers to neglecting the continuous assessment of changes in statistical characteristics or data distributions, which is crucial for maintaining model performance.
Data drift occurs when the incoming data distribution differs from the training data, leading to degraded model accuracy over time. Cloud providers recommend implementing skew and drift detection mechanisms to monitor these changes and alert developers when significant changes occur. By detecting data drift early, models can be retrained or adjusted to ensure they continue performing as expected in production environments.

Your task is to refactor the code to eliminate the "Ignoring monitoring for data drift" misuse, while preserving the original behavior.

Be sure the refactored code follows best practices, is modular, and improves maintainability.

After refactoring, briefly list exactly what changes you made to fix the misuse.

Code Snippet:

from dotenv import load_dotenv
import openai
import os

load_dotenv()

#  azure connection
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_key = "open_api_key"
openai.api_base = "endpoint"
deployment_name='deployment_name'

def get_completion(system_message, user_message, deployment_name='deployment_name', temperature=0, max_tokens=1000) -> str:

    This method calls openai chatcompletion with the provided system message
    and user message(passed by user) and returns the content response returned
    by openai model.

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{user_message}"}
    ]

    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message["content"]

Please structure your response as follows:

Refactored Code:
[Refactored code]

Summary of Changes:
[Bullet-point list describing exactly what was changed]
"""

response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'llama3',
        'prompt': prompt,
        'stream': False
    }
)

result = response.json()['response']

# Save the result to a text file
output_file = "refactored_code_outpu.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result)

print(f"Response saved to {output_file}")