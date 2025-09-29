# Imports
import re
from dotenv import load_dotenv
from groq import Groq
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")  # Automatically read from .env
groq = Groq(api_key=groq_api_key)

# ---- FUNCTION TO REFACTOR/CLEAN CODE USING LLM ----
def refactor_code(code):
    """
    Refactor and clean a Python code snippet using Groq LLM.
    """
    prompt = f"""
        You are an AI assistant specialized in Python code quality.

        Task:
        1. Refactor the following Python code to follow best practices:
        - Use meaningful function and variable names
        - Add type hints
        - Improve efficiency
        - Add proper error handling
        - Write a clear docstring

        2. STRICT INSTRUCTIONS:
        - Do NOT include any reasoning, thoughts, or <think> tags.
        - Do NOT explain your steps or comment on your process.
        - Output ONLY these two sections:

        Refactored Code:
        <your refactored code here>

        Summary of Changes:
        - Bullet 1: Benefit
        - Bullet 2: Benefit

        Code:
        ```python
        {code}
    """
    try:
        response = groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-r1-distill-llama-70b",
            temperature=0
        )
        refactored_code = response.choices[0].message.content
        return refactored_code
    except Exception as e:
        print(f"[Refactor Error] {e}")
        return code  # Return original code if error occurs

# ---- EXAMPLE USAGE ----
if __name__ == "__main__":
    # New example code snippet for demonstration
    example_code = '''
def process_data(data):
    """
    Function to normalize a list of numbers.
    """
    normalized = []
    for i in data:  # Loop through each element
        normalized.append(i / max(data))  # Normalize by max value
    return normalized
    '''

    refactored = refactor_code(example_code)
    print("Original Code:\n", example_code)
    print("\nRefactored Code:\n", refactored)

