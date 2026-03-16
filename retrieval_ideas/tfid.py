import numpy as np
from scipy.spatial.distance import cosine
import re
from collections import Counter
import math


class TFIDFRetriever:
    """TF-IDF retrieval from scratch using numpy."""

    def __init__(self):
        self.vocab = []
        self.word_to_idx = {}
        self.idf = None
        self.tfidf_matrix = None
        self.documents = []

    # ----------------------------
    # Tokenization
    # ----------------------------
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    # ----------------------------
    # TF-IDF Demo Explanation
    # ----------------------------
    def tfidf_demo(self, number_of_documents=1000,
                   rare_term_count=2,
                   frequent_word_count=950):

        print(math.log(number_of_documents / (rare_term_count + 1)))
        print(math.log(number_of_documents / (frequent_word_count + 1)))

        rare_tf = rare_term_count / number_of_documents
        freq_tf = frequent_word_count / number_of_documents

        print(rare_tf)
        print(freq_tf)

        rare_tfidf = rare_tf * math.log(number_of_documents / (rare_term_count + 1))
        freq_tfidf = freq_tf * math.log(number_of_documents / (frequent_word_count + 1))

        print("tfidf for rare term:", rare_tfidf)
        print("tfidf for frequent term:", freq_tfidf)

    # ----------------------------
    # Fit TF-IDF
    # ----------------------------
    def fit(self, documents: list[str]):
        self.documents = documents

        tokenized = [self.tokenize(doc) for doc in documents]

        all_words = set()
        for tokens in tokenized:
            all_words.update(tokens)

        self.vocab = sorted(all_words)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

        n_docs = len(documents)
        n_terms = len(self.vocab)

        tf = np.zeros((n_docs, n_terms))

        for doc_idx, tokens in enumerate(tokenized):
            counts = Counter(tokens)
            for word, count in counts.items():
                tf[doc_idx, self.word_to_idx[word]] = count

        doc_lengths = tf.sum(axis=1, keepdims=True)
        doc_lengths[doc_lengths == 0] = 1
        tf = tf / doc_lengths

        doc_freq = (tf > 0).sum(axis=0)
        self.idf = np.log(n_docs / (doc_freq + 1)) + 1

        self.tfidf_matrix = tf * self.idf

        print(f"Vocabulary size: {n_terms}")
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    # ----------------------------
    # Query Vector
    # ----------------------------
    def _query_vector(self, query: str):

        tokens = self.tokenize(query)

        vec = np.zeros(len(self.vocab))
        counts = Counter(tokens)

        for word, count in counts.items():
            if word in self.word_to_idx:
                vec[self.word_to_idx[word]] = count

        total = vec.sum()
        if total > 0:
            vec = vec / total

        vec = vec * self.idf

        return vec

    # ----------------------------
    # Search
    # ----------------------------
    def search(self, query: str, top_k=3):

        q_vec = self._query_vector(query)

        scores = []

        for idx in range(len(self.documents)):
            d_vec = self.tfidf_matrix[idx]

            if np.linalg.norm(q_vec) == 0 or np.linalg.norm(d_vec) == 0:
                sim = 0.0
            else:
                sim = 1 - cosine(q_vec, d_vec)

            scores.append((idx, sim, self.documents[idx]))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    # ----------------------------
    # Inspect IDF
    # ----------------------------
    def show_idf(self, words):

        print("\nIDF scores:")
        for word in words:
            if word in self.word_to_idx:
                print(f"'{word}': {self.idf[self.word_to_idx[word]]:.3f}")

    # ----------------------------
    # Example Runner
    # ----------------------------
    def demo(self):

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

        self.fit(documents)

        print("\nQuery: 'cat mat'")
        for idx, score, doc in self.search("cat mat"):
            print(f"[{idx}] score={score:.3f} | {doc}")

        print("\nQuery: 'loyal friendly pets'")
        for idx, score, doc in self.search("loyal friendly pets"):
            print(f"[{idx}] score={score:.3f} | {doc}")

        self.show_idf(["the", "cat", "kitten", "loyal", "adorable"])


# ----------------------------
# Main
# ----------------------------

def main():
    tfidf = TFIDFRetriever()
    tfidf.tfidf_demo()
    tfidf.demo()


if __name__ == "__main__":
    main()

#Output:
# 5.809142990314028
# 0.05024121643674665
# 0.002
# 0.95
# tfidf for rare term: 0.011618285980628055
# tfidf for frequent term: 0.04772915561490931
# Vocabulary size: 35
# TF-IDF matrix shape: (8, 35)

# Query: 'cat mat'
# [0] score=0.523 | The cat sat on the mat
# [4] score=0.317 | The mat was soft and warm
# [1] score=0.177 | The dog chased the cat in the garden

# Query: 'loyal friendly pets'
# [3] score=0.803 | Dogs are loyal and friendly pets
# [7] score=0.165 | Cats and dogs are the most popular pets worldwide
# [0] score=0.000 | The cat sat on the mat

# IDF scores:
# 'the': 1.288
# 'cat': 1.693
# 'kitten': 2.386
# 'loyal': 2.386
# 'adorable': 2.386