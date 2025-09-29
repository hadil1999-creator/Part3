import pandas as pd
from openai import OpenAI

# Initialize client
client = OpenAI()

# Load Excel file
df = pd.read_excel("preprocessing/updated_snippets.xlsx")

# Function to clean a code snippet using GPT
def clean_code(snippet):
    if pd.isna(snippet):  # Skip NaN values
        return None
    try:
        prompt = f"""
            You are given a code snippet.

            Your task:
            - Remove all explanatory comments, docstrings, and unnecessary text.
            - Do NOT modify or reformat the code itself.
            - Keep only:
            1. The actual code intact.
            2. Any line that contains "# File".
            3. Any line that matches the format "# ===== File: <filename> =====".

            Important:
            - Preserve the original formatting of the code and the kept lines.
            - Delete all other comments.

            Code snippet:
            {snippet}
            """
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can change to another available model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that cleans code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error processing snippet: {e}")
        return None

# Apply GPT cleaning to each snippet
df.loc[:0, "Cleaned_Code"] = df.loc[:0, "Code snippet"].apply(clean_code)


# Save back to Excel
df.to_excel("preprocessing/updated_snippets_cleaned.xlsx", index=False)

print("Cleaning complete! Saved to updated_snippets_cleaned.xlsx")
