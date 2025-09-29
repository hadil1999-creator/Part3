import pandas as pd
import time
from dotenv import load_dotenv
from groq import Groq
import os
import re

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Update your .env file with a valid key.")
else:
    print(groq_api_key)

groq = Groq(api_key=groq_api_key)

# === Load the data ===
df = pd.read_excel("training_checkpoint/instances_training_checkpoint.xlsx")

# === Prompt template ===
prompt_template =  """
You are a senior software engineer specializing in code quality, refactoring, and design patterns.

I will provide you with a code snippet that contains the "Not using training checkpoints" misuse.

Here is a description of the misuse:
Cloud providers offer the ability to resume training from the most recent checkpoint, saving the current state of the experiment rather than starting from scratch. This can save significant time and computational resources, especially when training large and complex models. However, developers may neglect to save training checkpoints in cloud storage. If a model fails and checkpoints have not been saved, the entire training job or pipeline will terminate, resulting in a loss of data since the model's state is not preserved in memory.
Your task is to refactor the code to eliminate the "Not using training checkpoints" misuse, while preserving the original behavior.

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
model_name = "deepseek-r1-distill-llama-70b"  # You can try other Groq models here

# === Output file names ===
output_file = "training_checkpoint/refactored_code_outputs_deepseek.txt"
timing_file = "training_checkpoint/model_timings_deepseek.txt"

# === Cumulative time tracker ===
cumulative_time = 0

# === Loop through examples ===
for idx, row in df.iterrows():
    row_start_time = time.time()

    try:
        code = row["Code snippet"]
        repo = row["Repository"]
        file = row["File"]

        prompt = prompt_template.format(code_snippet=code)

        response = groq.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        # === Save output ===
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"### Row: {idx+1}\n")
            f.write(f"### Repository: {repo}\n")
            f.write(f"### File: {file}\n")
            f.write(result + "\n\n")

        print(f"✅ Processed row {idx+1}/{len(df)}")

    except Exception as e:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"### Row: {idx+1}\n")
            f.write(f"### Repository: {repo}\n")
            f.write(f"### File: {file}\n")
            f.write(f"[ERROR] Could not process this row: {str(e)}\n\n")

        print(f"❌ Error in row {idx+1}: {str(e)} (continuing...)")

    # === Timing per row ===
    row_end_time = time.time()
    row_duration = round(row_end_time - row_start_time, 2)
    cumulative_time += row_duration

    with open(timing_file, "a", encoding="utf-8") as f:
        f.write(f"Row: {idx+1}, Repo: {repo}, File: {file}, Duration: {row_duration} sec\n")
        f.write(f"  → Cumulative Time: {cumulative_time} sec\n\n")

print(f"\n⏱️ Final total time for model '{model_name}': {cumulative_time} seconds")
print(f"⏳ Timing info saved to {timing_file}")
