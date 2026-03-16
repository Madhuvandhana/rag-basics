import numpy as np
import re
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
tfidf.fit(documents)


def tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def reciprocal_rank_fusion(ranked_lists, k=60, top_k=3):

    rrf_scores = {}
    doc_texts = {}

    for ranked_list in ranked_lists:
        for rank, (doc_idx, _, doc_text) in enumerate(ranked_list):

            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)
            doc_texts[doc_idx] = doc_text

    results = [(idx, score, doc_texts[idx]) for idx, score in rrf_scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]


def hybrid_search(query, documents, top_k=3):

    keyword_results = tfidf.search(query, top_k=len(documents))
    semantic_results = wv.embedding_search(query, documents, top_k=len(documents))

    return reciprocal_rank_fusion([keyword_results, semantic_results], top_k=top_k)

def compare_methods(query: str, top_k: int = 3):
    print(f"\n=== Query: '{query}' ===")

    print("TFIDF keyword results:")
    for idx, score, doc in tfidf.search(query, top_k=top_k):
        print(f"  [{idx}] score={score:.3f} | {doc}")

    print("\nEmbedding semantic results:")
    for idx, score, doc in wv.embedding_search(query, documents, top_k=top_k):
        print(f"  [{idx}] score={score:.3f} | {doc}")

    print("\nHybrid RRF results:")
    for idx, score, doc in hybrid_search(query, documents, top_k=top_k):
        print(f"  [{idx}] score={score:.4f} | {doc}")

def main():

    query = "puppy"

    print(f"Query: '{query}'")
    print("(Note: the word 'puppy' does NOT appear in any document)\n")

    print("TF-IDF (keyword) results:")

    kw_results = tfidf.search(query, top_k=3)

    for idx, score, doc in kw_results:
        print(f"[{idx}] score={score:.3f} | {doc}")

    if not any(s > 0 for _, s, _ in kw_results):
        print("(all scores are 0 — keyword search can't find 'puppy')")

    print("\nEmbedding (semantic) results:")

    for idx, score, doc in wv.embedding_search(query, documents, top_k=3):
        print(f"[{idx}] score={score:.3f} | {doc}")

    print("\nHybrid (TF-IDF + Embedding via RRF):")

    for idx, score, doc in hybrid_search(query, documents, top_k=3):
        print(f"[{idx}] score={score:.4f} | {doc}")


if __name__ == "__main__":
    main()
    compare_methods("cat mat")
    compare_methods("puppy")

#Output:
# Query: 'puppy'
# (Note: the word 'puppy' does NOT appear in any document)

# TF-IDF (keyword) results:
# (all scores are 0 — keyword search can't find 'puppy')

# Embedding (semantic) results:
# [5] score=0.981 | Puppies and kittens are adorable baby animals
# [2] score=0.978 | A kitten is a young cat
# [7] score=0.811 | Cats and dogs are the most popular pets worldwide

# Hybrid (TF-IDF + Embedding via RRF):
# [5] score=0.0164 | Puppies and kittens are adorable baby animals
# [2] score=0.0161 | A kitten is a young cat
# [7] score=0.0159 | Cats and dogs are the most popular pets worldwide


#Output:
# === Query: 'cat mat' ===
# TFIDF keyword results:

# Embedding semantic results:
#   [0] score=0.910 | The cat sat on the mat
#   [1] score=0.798 | The dog chased the cat in the garden
#   [7] score=0.790 | Cats and dogs are the most popular pets worldwide

# Hybrid RRF results:
#   [0] score=0.0164 | The cat sat on the mat
#   [1] score=0.0161 | The dog chased the cat in the garden
#   [7] score=0.0159 | Cats and dogs are the most popular pets worldwide