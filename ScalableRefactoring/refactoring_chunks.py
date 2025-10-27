import pandas as pd
import requests
import time
import json
import os
import ast

# === Load the data ===
df = pd.read_excel("preprocessing/split_files/2_Ignoring Testing Schema Mismatch.xlsx")

# === Load misuse definitions ===
script_dir = os.path.dirname(os.path.abspath(__file__))
misuse_file = os.path.join(script_dir, "misuses.json")
with open(misuse_file, "r", encoding="utf-8") as f:
    misuses = json.load(f)

# === Model settings ===
model_name = "llama3"
cumulative_time = 0

# === Output setup ===
output_dir = "ScalableRefactoring"
os.makedirs(output_dir, exist_ok=True)
misuse_name = df['Misuse'].iloc[0]
final_output_excel = os.path.join(output_dir, f"refactored_results_final_function_chunked_{model_name}_{misuse_name}.xlsx")

# === Function to split code by function/class definitions using AST ===
def split_code_by_functions(code):
    """Split Python code into chunks by function or class boundaries."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [code]  # fallback if parsing fails

    lines = code.splitlines()
    chunks = []
    start = 0

    for node in tree.body:
        end_line = getattr(node, "end_lineno", None)
        if end_line:
            chunk = "\n".join(lines[start:end_line])
            chunks.append(chunk)
            start = end_line

    if start < len(lines):
        chunks.append("\n".join(lines[start:]))

    return chunks

# === Prompt template ===
prompt_template = """
You are a senior software engineer specializing in code quality, refactoring, and design patterns.

I will provide you with a code snippet that contains the "{misuse_name}" misuse.

Misuse Description:
{misuse_description}

Task:
Refactor the code to fix the misuse using the same Machine Learning (ML) service used in the original code (e.g., Azure ML, Amazon SageMaker, or Google Vertex AI).

Important Instructions:
- Do NOT change variable names.
- Do NOT merge, reorder, or restructure the code.
- Do NOT change formatting, comments, or any code unrelated to the misuse.
- Fix the misuse using the ML service’s native functionalities when applicable.
- Make only the minimum required changes.

Code Snippet:
{code_snippet}

Response Format:

Refactored Code:
[Provide the fully refactored code here, keeping all original structure intact]

Summary of Changes:
[Provide a concise bullet-point list describing exactly what was changed to fix the misuse]
"""

# === Main loop ===
for idx, row in df.iterrows():
    row_start_time = time.time()
    try:
        code = row['Cleaned Code']
        misuse_name = row['Misuse']

        if misuse_name not in misuses:
            raise ValueError(f"Misuse '{misuse_name}' not found in JSON definitions.")

        misuse_description = misuses[misuse_name]['description']

        # --- Split code by function/class boundaries ---
        code_chunks = split_code_by_functions(code)

        # --- Refactor each chunk ---
        full_refactored = ""
        for part_num, chunk in enumerate(code_chunks, 1):
            prompt = prompt_template.format(
                misuse_name=misuse_name,
                misuse_description=misuse_description,
                code_snippet=chunk
            )

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=300
            )

            result_json = response.json()
            part_response = result_json.get("response", "[No response]")
            full_refactored += f"\n# === Part {part_num} ===\n" + part_response

        # === Save results ===
        df.at[idx, "Refactored_Code"] = full_refactored
        row_end_time = time.time()
        row_duration = round(row_end_time - row_start_time, 2)
        df.at[idx, "Row_Duration_sec"] = row_duration
        cumulative_time += row_duration

        print(f"✅ Processed row {idx+1}/{len(df)} ({misuse_name}) - Duration: {row_duration}s")

    except Exception as e:
        df.at[idx, "Refactored_Code"] = f"[ERROR] {str(e)}"
        row_end_time = time.time()
        row_duration = round(row_end_time - row_start_time, 2)
        df.at[idx, "Row_Duration_sec"] = row_duration
        cumulative_time += row_duration
        print(f"❌ Error in row {idx+1}: {e}")

    # --- Save progress incrementally ---
    df.to_excel(final_output_excel, index=False)

print(f"\n⏱️ Total runtime: {cumulative_time:.2f}s")
print(f"📄 Results saved to: {final_output_excel}")
