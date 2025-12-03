import pandas as pd

import json
from dotenv import load_dotenv
from groq import Groq
import os

# === Load the data ===
df = pd.read_excel("ScalableRefactoring/Refactoring_results_GPT.xlsx")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Update your .env file with a valid key.")
else:
    print('ok')

groq = Groq(api_key=groq_api_key)

# === Load misuse definitions ===
script_dir = os.path.dirname(os.path.abspath(__file__))
misuse_file = os.path.join(script_dir, "misuses.json")

with open(misuse_file, "r", encoding="utf-8") as f:
    misuses = json.load(f)


# === NEW PROMPT TEMPLATE: Validation only ===
prompt_template = """
You are an expert in cloud ML services (Azure, AWS, Google Cloud) and code quality.

Your task:
Evaluate whether the provided Refactored Code correctly fixes the misuse described below.

You MUST answer strictly one of the following formats:
- Yes
- No: <very short explanation>

Do NOT rewrite any code.

Misuse Name:
{misuse_name}

Misuse Definition:
{misuse_description}


Refactored Code:
{refactored_code}

Question:
Does the Refactored Code properly fix the misuse according to the misuse definition?
"""


# === Model settings ===
model_name = 'openai/gpt-oss-120b'

# Add output column
df['Refactoring_Valid'] = ""


# Ensure output folder exists
output_dir = "ScalableRefactoring"
os.makedirs(output_dir, exist_ok=True)

misuse_name = df['Misuse'].iloc[0]  # needed for file name

safe_model_name = model_name.replace("/", "_")

# Final output file
final_output_excel = os.path.join(
    output_dir,
    f"validation_results_{safe_model_name}.xlsx"
)


# === Loop through examples ===
for idx, row in df.iterrows():

    try:
        repo = row['Repository']
        file = row['File']
        misuse_name = row['Misuse']
        refactored_code = row['Refactored_Code']

        if misuse_name not in misuses:
            raise ValueError(f"Misuse '{misuse_name}' not found in JSON definitions.")

        misuse_description = misuses[misuse_name]['description']

        # Prepare validation prompt
        prompt = prompt_template.format(
            misuse_name=misuse_name,
            misuse_description=misuse_description,
            refactored_code=refactored_code
        )

        response = groq.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        # Save results
        df.at[idx, 'Refactoring_Valid'] = result

    except Exception as e:
        df.at[idx, 'Refactoring_Valid'] = f"[ERROR] {str(e)}"
        print(f"❌ Error in row {idx+1}: {str(e)} (continuing...)")

    # Save incrementally
    df.to_excel(final_output_excel, index=False)

print(f"📄 Validation results saved to {final_output_excel}")
