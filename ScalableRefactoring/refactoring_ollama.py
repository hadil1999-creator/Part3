import pandas as pd
import requests
import time
import json
import os

# === Load the data ===
df = pd.read_excel("preprocessing/split_files/3_Improper Handling Of Ml Api Limits.xlsx")


# === Load misuse definitions ===
script_dir = os.path.dirname(os.path.abspath(__file__))
misuse_file = os.path.join(script_dir, "misuses.json")

with open(misuse_file, "r", encoding="utf-8") as f:
    misuses = json.load(f)


# === Prompt template inside Python ===

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
- Fix the misuse using the ML service’s native functionalities when applicable (e.g., early stopping, checkpointing, logging, monitoring).
- Make only the minimum required changes to fix the misuse.
- Preserve the same ML service or SDK as in the original snippet.

Code Snippet:

{code_snippet}

Response Format:

Refactored Code:
[Provide the fully refactored code here, keeping all original structure intact]

Summary of Changes:
[Provide a concise bullet-point list describing exactly what was changed to fix the misuse]
"""


"""prompt_template = 
You are a senior software engineer specializing in code quality, refactoring, and design patterns.

I will provide you with a code snippet that contains the "{misuse_name}" misuse.

Here is a description of the misuse:
{misuse_description}

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

# === Model settings ===
model_name = 'gemma'
cumulative_time = 0

# Add new columns to store results
df['Refactored_Code'] = ""
df['Row_Duration_sec'] = 0.0

# Ensure output folder exists
output_dir = "ScalableRefactoring"
os.makedirs(output_dir, exist_ok=True)

# === Final output file ===
misuse_name = df['Misuse'].iloc[0]  # first row's misuse
final_output_excel = os.path.join(output_dir, f"refactored_results_final_changed_prompt_{model_name}_{misuse_name}.xlsx")

# === Loop through examples ===
for idx, row in df.iterrows():
    row_start_time = time.time()
    try:
        code = row['Cleaned Code']
        repo = row['Repository']
        file = row['File']
        misuse_name = row['Misuse']

        if misuse_name not in misuses:
            raise ValueError(f"Misuse '{misuse_name}' not found in JSON definitions.")

        misuse_description = misuses[misuse_name]['description']

        # Prepare prompt
        prompt = prompt_template.format(
            misuse_name=misuse_name,
            misuse_description=misuse_description,
            code_snippet=code
        )

        # Call the local API
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': model_name, 'prompt': prompt, 'stream': False}
        )

        result = response.json().get('response', '[No response]')

        # Save results in DataFrame
        df.at[idx, 'Refactored_Code'] = result
        row_end_time = time.time()
        row_duration = round(row_end_time - row_start_time, 2)
        df.at[idx, 'Row_Duration_sec'] = row_duration
        cumulative_time += row_duration

        print(f"✅ Processed row {idx+1}/{len(df)} ({misuse_name}) - Duration: {row_duration} sec")

    except Exception as e:
        df.at[idx, 'Refactored_Code'] = f"[ERROR] Could not process this row: {str(e)}"
        row_end_time = time.time()
        row_duration = round(row_end_time - row_start_time, 2)
        df.at[idx, 'Row_Duration_sec'] = row_duration
        cumulative_time += row_duration
        print(f"❌ Error in row {idx+1}: {str(e)} (continuing...)")

    # === Save after each iteration using the final filename ===
    df.to_excel(final_output_excel, index=False)

print(f"\n⏱️ Total time for model '{model_name}': {cumulative_time} seconds")
print(f"📄 Refactored results saved to {final_output_excel}")