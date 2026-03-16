import numpy as np
from scipy.spatial.distance import cosine
import re
from collections import Counter


#https://colab.research.google.com/drive/1UucgswVBAJ-r2hV5nLQV20NnrrnS-vdM?usp=sharing#scrollTo=8nndjgxq1ld
 # Tiny corpus about animals and simple descriptions
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
    """Simple lowercase tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())

def exact_match_search(query: str, docs: list[str]) -> list[str]:
    """Very naive baseline: only returns docs containing the exact string."""
    q = query.lower().strip()
    return [doc for doc in docs if q in doc.lower()]

print(f"Corpus size: {len(documents)} documents")
for i, doc in enumerate(documents):
    print(f"  [{i}] {doc}")

print("\nQuick baseline check (exact match only):")
for query in ["puppy", "kitten", "young cat"]:
    matches = exact_match_search(query, documents)
    print(f"  Query '{query}' -> {len(matches)} exact match(es)")


#Levenshtein helps with typos, but it does not understand meaning. The words "dog" and "puppy" are close in meaning, but not in character edits.
def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings using dynamic programming."""
    m, n = len(s1), len(s2)

    # Create a matrix of size (m+1) x (n+1)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Base cases: transforming empty string to/from a prefix
    for i in range(m + 1):
        dp[i][0] = i  # delete all chars from s1
    for j in range(n + 1):
        dp[0][j] = j  # insert all chars into s1

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # characters match, no edit needed
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    return dp[m][n]

# Let's see it in action
pairs = [("cat", "cat"), ("cat", "cta"), ("cat", "cats"), ("kitten", "sitting"), ("dog", "cat")]

for a, b in pairs:
    dist = levenshtein_distance(a, b)
    print(f"  '{a}' -> '{b}' = {dist} edit(s)")


def levenshtein_search(query: str, documents: list[str], top_k: int = 3) -> list[tuple[int, float, str]]:
    """Search documents by finding the closest Levenshtein match for each query word.

    For each word in the query, we find the minimum edit distance to any word
    in each document. Lower total distance = better match.
    """
    query_words = tokenize(query)
    scores = []

    for idx, doc in enumerate(documents):
        doc_words = tokenize(doc)
        total_distance = 0

        for qw in query_words:
            # Find the closest word in the document
            min_dist = min(levenshtein_distance(qw, dw) for dw in doc_words)
            total_distance += min_dist

        # Normalize by number of query words (lower is better)
        avg_dist = total_distance / len(query_words)
        scores.append((idx, avg_dist, doc))

    # Sort by distance (ascending — lower is better)
    scores.sort(key=lambda x: x[1])
    return scores[:top_k]

# Test: exact match
print("Query: 'cat mat'")
for idx, dist, doc in levenshtein_search("cat mat", documents):
    print(f"  [{idx}] dist={dist:.2f} | {doc}")

# Test: typo in query
print("\nQuery: 'cta' (typo for 'cat')")
for idx, dist, doc in levenshtein_search("cta", documents):
    print(f"  [{idx}] dist={dist:.2f} | {doc}")