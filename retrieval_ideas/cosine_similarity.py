import numpy as np
from scipy.spatial.distance import cosine
import re
from collections import Counter
import math


# TFIDF is stronger than raw overlap, but still limited for synonyms.
def cosine_similarity(vector_a, vector_b):
    """cos(θ) = (a·b) / (||a|| × ||b||)"""

    dot_product = np.dot(vector_a, vector_b)

    length_of_a = np.linalg.norm(vector_a)
    length_of_b = np.linalg.norm(vector_b)

    if length_of_a == 0 or length_of_b == 0:
        return 0.0

    return dot_product / (length_of_a * length_of_b)


def main():
    # Toy cosine similarity — 2D space: [cat-ness, dog-ness]
    # Each number = how much the doc is "about" that topic (0 to 1)

    query_vector = np.array([1.0, 0.0])           # Query: "cats"
    document_all_cats = np.array([1.0, 0.0])      # 100% cats
    document_all_dogs = np.array([0.0, 1.0])      # 100% dogs
    document_half = np.array([1.0, 1.0])          # 50% cats / 50% dogs

    print("Query [1, 0] = 'cats only'\n")

    print(
        f"vs all-cats doc [1,0]:  cos = {cosine_similarity(query_vector, document_all_cats):.2f}  (identical)"
    )

    print(
        f"vs all-dogs doc [0,1]:  cos = {cosine_similarity(query_vector, document_all_dogs):.2f}  (orthogonal)"
    )

    print(
        f"vs half-half [1,1]:     cos = {cosine_similarity(query_vector, document_half):.2f}  (partial match)"
    )

    print("\nCosine ignores length — only direction matters.")


if __name__ == "__main__":
    main()

#Output:
# Query [1, 0] = 'cats only'

#   vs all-cats doc [1,0]:  cos = 1.00  (identical)
#   vs all-dogs doc [0,1]:  cos = 0.00  (orthogonal)
#   vs half-half [1,1]:     cos = 0.71  (partial match)