import pandas as pd
import os
import re


def clean_misuse_name(name: str) -> str:
    """Normalize misuse names by removing newlines, extra spaces, and numbering artifacts."""
    if not isinstance(name, str):
        return name
    name = name.replace("\n", " ")  # Remove line breaks
    name = re.sub(r"\(\d+\)", "", name)  # Remove things like (3), (5)
    name = name.strip()  # Trim spaces
    name = re.sub(r"\s+", " ", name)  # Replace multiple spaces with one
    name = name.lower().strip()  # Normalize case to lowercase
    return name


def to_title_case(text: str) -> str:
    """Convert normalized misuse names to a readable title-case version."""
    return text.title()


def split_excel_by_misuse(input_file: str, output_dir: str = "preprocessing/split_files") -> None:
    df = pd.read_excel(input_file)

    if "Misuse" not in df.columns:
        raise ValueError("❌ The column 'Misuse' is missing in the Excel file.")

    # === Clean and normalize the Misuse column ===
    df["Misuse"] = df["Misuse"].apply(clean_misuse_name)
    df["Misuse"] = df["Misuse"].apply(to_title_case)

    # === Create output directory ===
    os.makedirs(output_dir, exist_ok=True)

    # === Get unique misuses ===
    misuses = sorted(df["Misuse"].dropna().unique())

    print(f"\n🔍 Found {len(misuses)} unique cleaned misuse types:\n")
    for i, misuse in enumerate(misuses, start=1):
        print(f"{i}. {misuse}")

    print("\n--- Generating split files ---\n")

    for i, misuse in enumerate(misuses, start=1):
        subset = df[df["Misuse"] == misuse]
        output_path = os.path.join(output_dir, f"{i}_{misuse}.xlsx")
        subset.to_excel(output_path, index=False)
        print(f"✅ [{i}] Saved {len(subset)} rows for '{misuse}' → {output_path}")

    print("\n🎯 All misuse files have been generated successfully!")


if __name__ == "__main__":
    input_file_path = "preprocessing/LLM_Preprocessed.xlsx"
    split_excel_by_misuse(input_file_path)
