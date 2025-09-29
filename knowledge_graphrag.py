# pip install graphrag
import json
from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

# === 0. Azure OpenAI credentials ===
from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings
from neo4j_graphrag.llm import AzureOpenAILLM

import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")



# Embeddings
embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model="text-embedding-3-small",
    api_version=API_VERSION
)

# LLM
llm = AzureOpenAILLM(
    model_name="gpt-4o",  # positional argument required
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=API_VERSION
)


# === 2. Initialize retriever and GraphRAG ===
retriever = VectorRetriever(embeddings)
gr = GraphRAG(retriever=retriever, llm=llm)

# === 3. Sample code to analyze ===
sample_code_1 = """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def calculate_total(prices):
    total = 0
    for p in prices:
        total += p
    return total
"""

sample_code_2 = """
def normalize(data):
    mean = sum(data) / len(data)
    return [(x - mean) for x in data]

def calculate_average(data):
    return sum(data) / len(data)
"""

# === 4. Add code to GraphRAG ===
gr.add_text_unit(sample_code_1, metadata={"type": "code"})
gr.add_text_unit(sample_code_2, metadata={"type": "code"})

# === 5. Build the graph ===
gr.build_graph()

# === 6. Summarize communities (clusters of related functions) ===
summaries = gr.summarize_communities()
print("\n=== COMMUNITY SUMMARIES ===")
print(json.dumps(summaries, indent=2))

# === 7. Ask for refactoring suggestions ===
query = """
Suggest refactoring opportunities:
- Merge duplicate or similar functions
- Break down long functions into smaller ones
- Improve code structure if possible
"""
refactoring_suggestions = gr.query(query)

print("\n=== REFACTORING SUGGESTIONS ===")
print(refactoring_suggestions)
