import pandas as pd
import ollama
import os

# File paths
input_file = "preprocessing/updated_snippets.xlsx"
output_file = "preprocessing/updated_snippets_cleaned_total.xlsx"

# Load Excel file 
if os.path.exists(output_file):
    df = pd.read_excel(output_file)  # resume from previously saved file
else:
    df = pd.read_excel(input_file)
    df["Cleaned_Code"] = None  # add column if not exists

# Function to clean a code snippet using LLaMA 3
def clean_code(snippet):
    if pd.isna(snippet):  # Skip NaN values
        return None
    try:
        prompt = f"""
        You are given a Python code snippet.

        Your task:
        - Remove only explanatory comments, docstrings, and any unnecessary text.
        - Keep all actual code intact, including:
        1. All imports
        2. All functions and classes
        3. All existing code lines
        4. Any line that contains "# File" or lines in the format "# ===== File: <filename> ====="
        - Do NOT:
        - Delete, rename, reorder, or move any functions, classes, imports, or code lines.
        - Add anything that is not already in the snippet.
        - Add headers, explanations, markdown, or any extra text.
        - Duplicate lines or invent code.
        - Only remove comments and docstrings, keeping the code exactly in the same order and structure.

        Code snippet:
        {snippet}
        """
        # Send prompt to LLaMA 3
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()

    except Exception as e:
        print(f"Error processing snippet: {e}")
        return None

# Find the last processed row index
last_processed = df[df["Cleaned_Code"].notna()].index.max()

# Start from the next unprocessed row
for idx, row in df.iloc[last_processed + 1:].iterrows():
    if pd.isna(row["Cleaned_Code"]):
        df.at[idx, "Cleaned_Code"] = clean_code(row["Code snippet"])
        print(f"✅ Processed row {idx + 1}/{len(df)}")
        df.to_excel(output_file, index=False)

print(f"Cleaning complete! Saved to {output_file}")
