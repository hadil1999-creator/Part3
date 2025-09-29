import pandas as pd
import requests
import time

# === Load the data ===
df = pd.read_excel("data_drift/instances_data_drift.xlsx")

# === Prompt template ===
prompt_template = """
You are a senior software engineer specializing in code quality, refactoring, and design patterns.

I will provide you with a code snippet that contains the "Ignoring monitoring for data drift" misuse.

Here is a description of the misuse:
it refers to neglecting the continuous assessment of changes in statistical characteristics or data distributions, which is crucial for maintaining model performance.
Data drift occurs when the incoming data distribution differs from the training data, leading to degraded model accuracy over time. Cloud providers recommend implementing skew and drift detection mechanisms to monitor these changes and alert developers when significant changes occur. By detecting data drift early, models can be retrained or adjusted to ensure they continue performing as expected in production environments.

Your task is to refactor the code to eliminate the "Ignoring monitoring for data drift" misuse, while preserving the original behavior.

Be sure the refactored code follows best practices, is modular, and improves maintainability.

Do not make any other changes except those required to fix the described behavior.

After refactoring, briefly list exactly what changes you made to fix the misuse.

Code Snippet:

{code_snippet}

Please structure your response as follows:

Refactored Code:
[Refactored code]

Summary of Changes:
[Bullet-point list describing exactly what was changed]
"""

# === Model name ===
model_name = 'codellama'  # change this when benchmarking another model

# === Output file names ===
output_file = "data_drift/refactored_code_data_drift_outputs1_codellama.txt"
timing_file = "data_drift/model_timings1_codellama.txt"

# === Cumulative time tracker ===
cumulative_time = 0

# === Loop through examples ===
for idx, row in df.iterrows(): #added this
    row_start_time = time.time()

    try:
        code = row['Code snippet']
        repo = row['Repository']
        file = row['File']
        
        prompt = prompt_template.format(code_snippet=code)

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model_name,
                'prompt': prompt,
                'stream': False
            }
        )

        result = response.json().get('response', '[No response]')

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"### Row: {idx+1}\n")
            f.write(f"### Repository: {repo}\n")
            f.write(f"### File: {file}\n")
            f.write(result.strip() + "\n\n")

        print(f"✅ Processed row {idx+1}/{len(df)}")

    except Exception as e:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"### Row: {idx+1}\n")
            f.write(f"### Repository: {repo}\n")
            f.write(f"### File: {file}\n")
            f.write(f"[ERROR] Could not process this row: {str(e)}\n\n")

        print(f"❌ Error in row {idx+1}: {str(e)} (continuing...)")

    row_end_time = time.time()
    row_duration = round(row_end_time - row_start_time, 2)
    cumulative_time += row_duration

    # === Save cumulative time after each row ===
    with open(timing_file, "a", encoding="utf-8") as f:
        f.write(f"Row: {idx+1}, Repo: {repo}, File: {file}, Duration: {row_duration} sec\n")
        f.write(f"  → Cumulative Time: {cumulative_time} sec\n\n")


print(f"\n⏱️ Final total time for model '{model_name}': {cumulative_time} seconds")
print(f"⏳ Timing info saved to {timing_file}")
 