import os
from dotenv import load_dotenv
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from contextlib import redirect_stdout
import time

# ---------------------------
# Step 0: Load environment variables
# ---------------------------
load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# ---------------------------
# Step 1: Example large codebase
# ---------------------------
large_code = """
import math

def add(a, b):
    return a + b

def multiply(a, b):
    result = 0
    for _ in range(b):
        result += a
    return result

class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x

    def subtract(self, x):
        self.value -= x

def big_function():
    data = [i for i in range(1000)]
    squared = [x**2 for x in data]
    total = sum(squared)
    return total
"""

# ---------------------------
# Step 2-6: Redirect all prints to a file
# ---------------------------
with open("refactoring_output.txt", "w", encoding="utf-8") as f:
    with redirect_stdout(f):

        # Step 2: Split the code using tokens
        token_splitter = TokenTextSplitter(
            chunk_size=10,      # number of tokens per chunk
            chunk_overlap=0     # overlap in tokens
        )
        token_chunks = token_splitter.split_text(large_code)
        print("Number of chunks:", len(token_chunks))
        documents = [Document(page_content=chunk) for chunk in token_chunks]
        print("documents:", documents)

        # Step 3: Create Azure embeddings
        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version="2024-02-01"
        )
        print("embeddings:", embeddings)

        # Step 4: FAISS vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        print("vector_store:", vector_store)

        # Step 5: Query relevant code
        query = "Refactor code related to dataset processing"
        docs = vector_store.similarity_search(query, k=2)
        print("docs:", docs)

        print("=== Retrieved Chunks ===")
        for i, doc in enumerate(docs, 1):
            print(f"Chunk {i}:\n{doc.page_content}\n{'-'*30}")

        # Step 6: Mock LLM refactoring
        def llm_refactor_code(code_chunk):
            return f"# Refactored version of:\n{code_chunk}"

        refactored_chunks = [llm_refactor_code(doc.page_content) for doc in docs]

        print("\n=== Refactored Chunks ===")
        for chunk in refactored_chunks:
            print(chunk)

print("✅ All output saved to refactoring_output.txt")
