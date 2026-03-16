import os
from getpass import getpass
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from sklearn.decomposition import PCA
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

# API Key Setup
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"


# -----------------------------------
# Embedding function
# -----------------------------------
def get_embedding(text, model=EMBEDDING_MODEL):
    text_clean = text.replace("\n", " ")
    resp = client.embeddings.create(
        input=[text_clean],
        model=model
    )
    return np.array(resp.data[0].embedding)


# -----------------------------------
# Token counter
# -----------------------------------
def num_tokens_from_string(s: str, encoding_name: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(s))


# -----------------------------------
# Main demo
# -----------------------------------
def main():

    sample = "The quick brown fox jumps over the lazy dog."

    emb = get_embedding(sample)

    print(f"Vector length: {len(emb)}")
    print("First 5 dims:", emb[:5])

    print("Sample token count:", num_tokens_from_string(sample))

    sentences = [
        "I love machine learning",
        "OpenAI creates powerful AI models",
        "The sky is clear today",
        "I enjoy hiking in the mountains",
        "This restaurant has great food",
    ]

    vectors = np.vstack([get_embedding(s) for s in sentences])

    # PCA to 2D
    pca = PCA(n_components=2)
    points = pca.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1])

    for i, txt in enumerate(sentences):
        plt.annotate(txt, (points[i, 0], points[i, 1]))

    plt.title("2D PCA of Sentence Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

def compute_similarity(sentences):
    """Embed sentences and compute cosine similarity matrix."""
    
    vectors = np.vstack([get_embedding(s) for s in sentences])

    similarity = cosine_similarity(vectors)

    print("Cosine Similarity Matrix:\n")
    print(similarity)

    return vectors, similarity


def find_most_similar(sentences, similarity):
    """Find the two most similar sentences."""

    sim_copy = similarity.copy()

    # Ignore self-similarity
    np.fill_diagonal(sim_copy, -1)

    i, j = np.unravel_index(np.argmax(sim_copy), sim_copy.shape)

    print("\nMost similar sentences:")
    print(sentences[i])
    print(sentences[j])
    print("Similarity score:", sim_copy[i][j])

    return i, j


def visualize_embeddings(sentences, vectors):
    """Visualize embeddings using PCA."""

    pca = PCA(n_components=2)
    points = pca.fit_transform(vectors)

    plt.figure(figsize=(8,6))
    plt.scatter(points[:,0], points[:,1])

    for k, txt in enumerate(sentences):
        plt.annotate(txt, (points[k,0], points[k,1]))

    plt.title("Sentence Embedding Visualization (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


#For accurate similarity comparisons, always use cosine similarity on the full vectors (see below)
def run_similarity_demo():

    sentences = [
        "I love studying artificial intelligence",
        "Machine learning is fascinating",
        "The weather is sunny today",
        "AI systems learn from data",
        "I enjoy hiking in nature"
    ]

    vectors, similarity = compute_similarity(sentences)

    find_most_similar(sentences, similarity)

    visualize_embeddings(sentences, vectors)
    # Compare two specific sentences
    similarity_0_1 = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))
    print(f"Similarity between sentence 0 and 1: {similarity_0_1[0][0]:.4f}")

#Output:
# Cosine Similarity Matrix:

# [[1.         0.57274763 0.09679466 0.49082101 0.23644727]
#  [0.57274763 1.         0.07901183 0.47666378 0.15929017]
#  [0.09679466 0.07901183 1.         0.07403804 0.23080072]
#  [0.49082101 0.47666378 0.07403804 1.         0.07120451]
#  [0.23644727 0.15929017 0.23080072 0.07120451 1.        ]]

# Most similar sentences:
# I love studying artificial intelligence
# Machine learning is fascinating
# Similarity score: 0.5727476294441121

#Similarity between sentence 0 and 1: 0.5727

if __name__ == "__main__":
    run_similarity_demo()