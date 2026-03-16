import numpy as np
from scipy.spatial.distance import cosine
import re
from collections import Counter
import math
import word_vectors as wv
import tfid

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

def weighted_rrf(
    ranked_lists: list[list[tuple[int, float, str]]],
    weights: list[float],
    k: int = 60,
    top_k: int = 3
) -> list[tuple[int, float, str]]:
    """RRF with per retriever weights."""
    if len(ranked_lists) != len(weights):
        raise ValueError("ranked_lists and weights must have the same length")

    fused_scores = {}
    doc_texts = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (doc_idx, _, doc_text) in enumerate(ranked_list):
            contribution = weight * (1 / (k + rank + 1))
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + contribution
            doc_texts[doc_idx] = doc_text

    results = [(idx, score, doc_texts[idx]) for idx, score in fused_scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def weighted_hybrid_search(
    query: str,
    documents: list[str],
    keyword_weight: float = 1.0,
    semantic_weight: float = 1.0,
    top_k: int = 3
) -> list[tuple[int, float, str]]:
    """Hybrid search with adjustable keyword and semantic weights."""
    keyword_results = tfidf.search(query, top_k=len(documents))
    semantic_results = wv.embedding_search(query, documents, top_k=len(documents))

    return weighted_rrf(
        ranked_lists=[keyword_results, semantic_results],
        weights=[keyword_weight, semantic_weight],
        top_k=top_k,
    )

query = "friendly kitten"
print(f"Query: '{query}'")

settings = [
    (1.0, 1.0, "Equal weights"),
    (2.0, 0.5, "Keyword heavy"),
    (0.5, 2.0, "Semantic heavy"),
]

for kw, sem, label in settings:
    print(f"\n{label} keyword={kw} semantic={sem}")
    for idx, score, doc in weighted_hybrid_search(query, documents, kw, sem, top_k=3):
        print(f"  [{idx}] score={score:.4f} | {doc}")

#Output:
# Query: 'friendly kitten'

# Equal weights keyword=1.0 semantic=1.0
#   [2] score=0.0164 | A kitten is a young cat
#   [5] score=0.0161 | Puppies and kittens are adorable baby animals
#   [7] score=0.0159 | Cats and dogs are the most popular pets worldwide

# Keyword heavy keyword=2.0 semantic=0.5
#   [2] score=0.0082 | A kitten is a young cat
#   [5] score=0.0081 | Puppies and kittens are adorable baby animals
#   [7] score=0.0079 | Cats and dogs are the most popular pets worldwide

# Semantic heavy keyword=0.5 semantic=2.0
#   [2] score=0.0328 | A kitten is a young cat
#   [5] score=0.0323 | Puppies and kittens are adorable baby animals
#   [7] score=0.0317 | Cats and dogs are the most popular pets worldwide