import re


# Jaccard is simple and fast. It treats every word equally and ignores word frequency
# and also misses semantic search.

def tokenize(text: str) -> list[str]:
    """Simple lowercase tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


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


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    intersection = set_a & set_b
    union = set_a | set_b

    if not union:
        return 0.0

    return len(intersection) / len(union)


def jaccard_search(query: str, documents: list[str], top_k: int = 3):
    """Rank documents by Jaccard similarity with the query."""

    query_set = set(tokenize(query))
    scores = []

    for idx, doc in enumerate(documents):
        doc_set = set(tokenize(doc))
        sim = jaccard_similarity(query_set, doc_set)
        scores.append((idx, sim, doc))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]


def main():

    # Demonstrate with individual set comparisons first
    q = set(tokenize("cat on the mat"))
    d0 = set(tokenize(documents[0]))
    d1 = set(tokenize(documents[1]))

    print(f"Query words:  {q}")
    print(f"Doc 0 words:  {d0}")
    print(f"Intersection: {q & d0}")
    print(f"Union:        {q | d0}")
    print(f"Jaccard:      {jaccard_similarity(q, d0):.3f}")

    print()

    print(f"Doc 1 words:  {d1}")
    print(f"Intersection: {q & d1}")
    print(f"Jaccard:      {jaccard_similarity(q, d1):.3f}")

    print("\nQuery: 'cat on the mat'")
    for idx, score, doc in jaccard_search("cat on the mat", documents):
        print(f"[{idx}] score={score:.3f} | {doc}")

    print("\nQuery: 'loyal friendly pets'")
    for idx, score, doc in jaccard_search("loyal friendly pets", documents):
        print(f"[{idx}] score={score:.3f} | {doc}")


if __name__ == "__main__":
    main()

#Output:
# Query words:  {'cat', 'mat', 'on', 'the'}
# Doc 0 words:  {'cat', 'mat', 'the', 'on', 'sat'}
# Intersection: {'cat', 'the', 'on', 'mat'}
# Union:        {'cat', 'mat', 'the', 'on', 'sat'}
# Jaccard:      0.800

# Doc 1 words:  {'cat', 'in', 'dog', 'chased', 'garden', 'the'}
# Intersection: {'cat', 'the'}
# Jaccard:      0.250

# Query: 'cat on the mat'
#   [0] score=0.800 | The cat sat on the mat
#   [1] score=0.250 | The dog chased the cat in the garden
#   [4] score=0.250 | The mat was soft and warm

# Query: 'loyal friendly pets'
#   [3] score=0.500 | Dogs are loyal and friendly pets
#   [7] score=0.091 | Cats and dogs are the most popular pets worldwide
#   [0] score=0.000 | The cat sat on the mat