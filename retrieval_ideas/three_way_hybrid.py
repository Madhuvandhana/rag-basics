import numpy as np
from scipy.spatial.distance import cosine
import re
from collections import Counter
import math
import word_vectors as wv
import tfid
import jaccard_similarity as js
import hybrid_retrieval

# Local embedding dictionary
WORD_VECTORS = {
    "cat":      np.array([0.9, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0]),
    "kitten":   np.array([0.9, 0.8, 0.9, 0.0, 0.3, 0.0, 0.0]),
    "dog":      np.array([0.9, 0.9, 0.0, 0.0, 0.1, 0.3, 0.0]),
    "puppy":    np.array([0.9, 0.9, 0.9, 0.0, 0.2, 0.3, 0.0]),
    "puppies":  np.array([0.9, 0.9, 0.9, 0.0, 0.2, 0.3, 0.0]),
    "kittens":  np.array([0.9, 0.8, 0.9, 0.0, 0.3, 0.0, 0.0]),
    "cats":     np.array([0.9, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0]),
    "dogs":     np.array([0.9, 0.9, 0.0, 0.0, 0.1, 0.3, 0.0]),
    "pets":     np.array([0.7, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "animals":  np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "baby":     np.array([0.3, 0.2, 1.0, 0.0, 0.4, 0.0, 0.0]),
    "young":    np.array([0.2, 0.1, 0.9, 0.0, 0.1, 0.0, 0.0]),
    "adorable": np.array([0.4, 0.5, 0.6, 0.0, 0.5, 0.0, 0.0]),
    "loyal":    np.array([0.3, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "friendly": np.array([0.3, 0.5, 0.0, 0.0, 0.2, 0.0, 0.0]),
    "sat":      np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.8, 0.0]),
    "chased":   np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0]),
    "mat":      np.array([0.0, 0.0, 0.0, 0.6, 0.8, 0.0, 0.0]),
    "garden":   np.array([0.0, 0.0, 0.0, 0.9, 0.2, 0.0, 0.7]),
    "flowers":  np.array([0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 1.0]),
    "birds":    np.array([0.8, 0.1, 0.0, 0.4, 0.0, 0.3, 0.2]),
    "soft":     np.array([0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0]),
    "warm":     np.array([0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0]),
    "popular":  np.array([0.1, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "worldwide":np.array([0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0]),
    "most":     np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "full":     np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.2]),
}

documents = [
    "The cat sat on the mat",
    "The dog chased the cat in the garden",
    "A kitten is a young cat",
    "Dogs are loyal and friendly pets",
    "The mat was soft and warm",
    "Puppies and kittens are adorable baby animals",
    "The garden was full of flowers and birds",
    "Cats and dogs are the most popular pets worldwide",
]
tfidf = tfid.TFIDFRetriever()

def three_way_hybrid(query: str, documents: list[str], top_k: int = 3) -> list[tuple[int, float, str]]:
    """Combine TFIDF, embedding, and Jaccard via RRF."""
    keyword_results = tfidf.search(query, top_k=len(documents))
    semantic_results = wv.embedding_search(query, documents, top_k=len(documents))
    jaccard_results = js.jaccard_search(query, documents, top_k=len(documents))

    return hybrid_retrieval.reciprocal_rank_fusion(
        [keyword_results, semantic_results, jaccard_results],
        top_k=top_k,
    )

query = "adorable baby cat"
print(f"Query: '{query}'")

print("\nTFIDF only:")
for idx, score, doc in tfidf.search(query, top_k=3):
    print(f"  [{idx}] score={score:.3f} | {doc}")

print("\nEmbedding only:")
for idx, score, doc in wv.embedding_search(query, documents, top_k=3):
    print(f"  [{idx}] score={score:.3f} | {doc}")

print("\nTwo way hybrid TFIDF plus embedding:")
for idx, score, doc in hybrid_retrieval.hybrid_search(query, documents, top_k=3):
    print(f"  [{idx}] score={score:.4f} | {doc}")

print("\nThree way hybrid plus Jaccard:")
for idx, score, doc in three_way_hybrid(query, documents, top_k=3):
    print(f"  [{idx}] score={score:.4f} | {doc}")

#Output
# Query: 'adorable baby cat'

# TFIDF only:

# Embedding only:
#   [5] score=0.987 | Puppies and kittens are adorable baby animals
#   [2] score=0.979 | A kitten is a young cat
#   [7] score=0.779 | Cats and dogs are the most popular pets worldwide

# Two way hybrid TFIDF plus embedding:
#   [5] score=0.0164 | Puppies and kittens are adorable baby animals
#   [2] score=0.0161 | A kitten is a young cat
#   [7] score=0.0159 | Cats and dogs are the most popular pets worldwide

# Three way hybrid plus Jaccard:
#   [5] score=0.0328 | Puppies and kittens are adorable baby animals
#   [2] score=0.0320 | A kitten is a young cat
#   [0] score=0.0315 | The cat sat on the mat