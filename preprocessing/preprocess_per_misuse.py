import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Update your .env file with a valid key.")

# Initialize Groq client
groq = Groq(api_key=groq_api_key)

# === Load the data ===
df = pd.read_excel("preprocessing/cleaned_code_python(colab).xlsx")



# === Function to query Groq ===
def extract_ml_api_relevant_code(code_snippet: str) -> str:
    prompt = f"""
You are preprocessing source code to prepare it for ML service misuse analysis.

Input code:
{code_snippet}

Task:
- Keep ONLY:
  • Lines that use or call ML services or ML APIs (e.g., API calls, client initialization, request handling).
  • Any helper functions, variables, or imports that are strictly required for those calls to work.
  • Existing file markers (lines starting with '# ===== File:').
- Preserve the original structure of the code (do not invent new functions, classes, or variables).
- Remove ALL unrelated or unused code.
- Do not add or remove file markers.
- Do not insert comments, explanations, or text such as 'deleted' or 'this will help'.
- Output must be valid Python code that reflects only the filtered parts.
- If nothing relevant remains in a file, keep only its file marker line.
"""

    try:
        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return code_snippet


# === Function to query Groq ===
def extract_data_drift_relevant_code(code_snippet: str) -> str:
    prompt = f"""
You are preprocessing source code to prepare it for ML service misuse analysis.

Input code:
{code_snippet}

Task:
- Keep ONLY:
  • Code related to data ingestion, data preprocessing, and model input/output pipelines.
  • Code related to monitoring, metrics collection, or evaluation of data/model performance.
  • Imports, helper functions, and variables strictly required for the above.
  • File markers (lines starting with '# ===== File:').
- Preserve the original structure of the code (do not invent new code).
- Remove ALL unrelated or unused code.
- Do not add or remove file markers.
- Do not insert explanations, placeholders, or comments.
- Output must be valid Python code containing only the filtered parts.
- If no relevant code remains in a file, keep only its file marker line.
"""

    try:
        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return code_snippet


# === Function to query Groq ===
def extract_schema_mismatch_relevant_code(code_snippet: str) -> str:
    prompt = f"""
You are preprocessing source code to prepare it for ML service misuse analysis.

Input code:
{code_snippet}

Task:
- Keep ONLY:
  • Code related to data ingestion/loading for training, testing, or production datasets.
  • Code that defines or validates schemas, feature sets, or data structure checks.
  • Code related to model training, evaluation, or testing pipelines that depend on schema alignment.
  • Imports, helper functions, and variables strictly required for the above.
  • File markers (lines starting with '# ===== File:').
- Preserve the original structure of the code (do not invent new code).
- Remove ALL unrelated or unused code.
- Do not add or remove file markers.
- Do not insert explanations, placeholders, or comments.
- Output must be valid Python code containing only the filtered parts.
- If no relevant code remains in a file, keep only its file marker line.
"""

    try:
        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return code_snippet



# === Apply only where needed with incremental saving ===
# Initialize the new column if it doesn't exist
if "LLM_Process" not in df.columns:
    df["LLM_Process"] = None

# Find last processed row to resume if needed
last_processed = df[df["LLM_Process"].notna()].index.max() if df["LLM_Process"].notna().any() else -1

# Start processing
for idx, row in df.iloc[last_processed + 1:].iterrows():
    if pd.isna(row["LLM_Process"]):
        processed = None  # default
        if row["Misuse"] == "Improper handling of ML API limits":
            processed = extract_ml_api_relevant_code(row["Cleaned Code"])
        elif row["Misuse"] == "Ignoring monitoring data drift":
            processed = extract_data_drift_relevant_code(row["Cleaned Code"])
        elif row["Misuse"] == " Ignoring testing schema mismatch":
            processed = extract_schema_mismatch_relevant_code(row["Cleaned Code"])
        
        df.at[idx, "LLM_Process"] = processed
        if processed is not None:
            print(f"✅ Processed row {idx + 1}/{len(df)} and saved")
            df.to_excel("preprocessing/LLM_Preprocessed.xlsx", index=False)
