import os
from getpass import getpass
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from sklearn.decomposition import PCA
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
import re

# https://colab.research.google.com/drive/1TnTWElFXTdU4RxGQtwICdf3POHKNjlm1?usp=sharing#scrollTo=yJ_EI1kqHF7Y
# API Key Setup
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

client = OpenAI()
MODEL = "gpt-5.2"
EMBEDDING_MODEL = "text-embedding-3-small"

#window_size: how many sentences per chunk
# stride: how many sentences to advance between chunks (a stride smaller than the window creates overlap)
#Idea is to chunk texts in order to reduce input context token cost by doing vector search on small chunk
def chunk_text(text, window_size=5, stride=2):
    """Split text into overlapping chunks using a sliding window over sentences."""
    sentences = re.split('(?<=[.!?]) +', text.strip()) # Split text into sentences based on punctuation
    chunks = []
    for i in range(0, len(sentences) - window_size + 1, stride): # Loop through the sentences and create chunks
        chunks.append(' '.join(sentences[i:i + window_size])) # Join the sentences into a chunk
    return chunks

# Load the article from a file instead of pasting it inline
with open("data/batman_history.md") as f:
    text = f.read()

chunks = chunk_text(text, window_size=4, stride=1)
print(f"Created {len(chunks)} chunks from the article\n")

# Preview the first chunk
print("Example chunk:")
print(chunks[0])

# def get_embedding(text: str, client: OpenAI) -> list[float]:
#     response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
#     return response.data[0].embedding

# vectors = np.array([get_embedding(chunk, client) for chunk in chunks])
# print(f"Embedded {vectors.shape[0]} chunks into {vectors.shape[1]}-dimensional vectors")

# # Build a FAISS index for fast similarity search
# index = faiss.IndexFlatL2(vectors.shape[1])
# index.add(vectors)
# print(f"FAISS index built with {index.ntotal} vectors")