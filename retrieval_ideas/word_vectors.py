import numpy as np
from scipy.spatial.distance import cosine
import re


# Use a combination of fuzzy search, word vectors and Levenshtein distance
word_vectors = {
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


def tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def embed_text(text: str, vectors: dict) -> np.ndarray:
    tokens = tokenize(text)
    vecs = [vectors[t] for t in tokens if t in vectors]

    if not vecs:
        return np.zeros(7)

    return np.mean(vecs, axis=0)


def embedding_search(query: str, documents: list[str], top_k: int = 3):
    q_emb = embed_text(query, word_vectors)
    scores = []

    for idx, doc in enumerate(documents):
        d_emb = embed_text(doc, word_vectors)

        if np.linalg.norm(q_emb) == 0 or np.linalg.norm(d_emb) == 0:
            sim = 0.0
        else:
            sim = 1 - cosine(q_emb, d_emb)

        scores.append((idx, sim, doc))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def main():

    print("Cosine similarity between word pairs:")

    pairs = [
        ("cat", "kitten"),
        ("cat", "dog"),
        ("cat", "garden"),
        ("puppy", "kitten"),
        ("mat", "soft"),
    ]

    for w1, w2 in pairs:
        sim = 1 - cosine(word_vectors[w1], word_vectors[w2])
        print(f"{w1:8s} <-> {w2:8s} = {sim:.3f}")

    print("\nQuery: 'puppy'")
    for idx, score, doc in embedding_search("puppy", documents):
        print(f"[{idx}] score={score:.3f} | {doc}")

    print("\nQuery: 'young animal'")
    for idx, score, doc in embedding_search("young animal", documents):
        print(f"[{idx}] score={score:.3f} | {doc}")


if __name__ == "__main__":
    main()


#Output:
# Cosine similarity between word pairs:
# cat      <-> kitten   = 0.807
# cat      <-> dog      = 0.968
# cat      <-> garden   = 0.028
# puppy    <-> kitten   = 0.978
# mat      <-> soft     = 0.861

# Query: 'puppy'
# [5] score=0.981 | Puppies and kittens are adorable baby animals
# [2] score=0.978 | A kitten is a young cat
# [7] score=0.811 | Cats and dogs are the most popular pets worldwide

# Query: 'young animal'
# [5] score=0.767 | Puppies and kittens are adorable baby animals
# [2] score=0.745 | A kitten is a young cat
# [7] score=0.229 | Cats and dogs are the most popular pets worldwide