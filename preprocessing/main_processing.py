"""Problems with the Current Prompts & LLM Preprocessing

Too strict filtering rules

If the LLM cannot confidently identify relevant lines, it removes everything, resulting in empty output.

LLM cannot reliably detect all relevant code, especially if it is indirectly related or abstracted.

Dependency analysis is hard for LLM

Code may have indirect dependencies (helper functions, variable initialization, utility code).

LLM often deletes these because they don’t explicitly match the task keywords. (hard for us to reassemble the code)

LLM may interpret “nothing relevant” as “delete everything.”

Original code context is lost

LLM processes the code as a single snippet, without understanding file structure, imports, or execution flow.

This causes the LLM to remove lines that are necessary for correctness but not explicitly mentioned in the prompt.

LLM hallucination / overconfidence

It tends to over-prune if unsure about relevance.

LLM may only focus on parts that mention keywords and ignore other relevant code.

If code is flattened or newlines/indentation are inconsistent, the LLM cannot parse structure properly.
"""

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import os
import json

# === Load environment variables ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Update your .env file with a valid key.")

# === Initialize Groq client ===
groq = Groq(api_key=groq_api_key)

# === Load the data ===
df = pd.read_excel("preprocessing/cleaned_code_python(colab).xlsx")

# === Load prompts from external file ===
with open("preprocessing/prompts.json", "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

# === Generalized function to query Groq ===
def extract_relevant_code(code_snippet: str, misuse_type: str) -> str:
    if misuse_type not in PROMPTS:
        raise ValueError(f"❌ Unknown misuse type: {misuse_type}")

    prompt = PROMPTS[misuse_type].format(code_snippet=code_snippet)

    try:
        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Error calling Groq API: {e}")
        return code_snippet


# === Apply only where needed with incremental saving ===
if "LLM_Process" not in df.columns:
    df["LLM_Process"] = None

# Find last processed row to resume if needed
last_processed = df[df["LLM_Process"].notna()].index.max() if df["LLM_Process"].notna().any() else -1

# Start processing
for idx, row in df.iloc[last_processed + 1:].iterrows():
    if pd.isna(row["LLM_Process"]):
        processed = None
        misuse = row["Misuse"].strip()

        if misuse == "Improper handling of ml api limits":
            processed = extract_relevant_code(row["Cleaned Code"], "Improper handling of ml api limits")

        elif misuse == "Ignoring monitoring data drift":
            processed = extract_relevant_code(row["Cleaned Code"], "Ignoring monitoring data drift")

        elif misuse == "Ignoring testing schema mismatch":
            processed = extract_relevant_code(row["Cleaned Code"], "Ignoring testing schema mismatch")
        
        df.at[idx, "LLM_Process"] = processed

        if processed is not None:
            print(f"✅ Processed row {idx + 1}/{len(df)} and saved")
            df.to_excel("preprocessing/LLM_Preprocessed.xlsx", index=False)
