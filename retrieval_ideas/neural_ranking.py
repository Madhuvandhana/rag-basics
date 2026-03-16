import numpy as np
from scipy.spatial.distance import cosine
import re
from collections import Counter
import math
import word_vectors as wordVector

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
    """Simple lowercase tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())

class ToyNeuralRanker:
    """A tiny neural network that scores query-document pairs.

    Architecture: concatenate query + doc embeddings -> hidden layer -> score
    This is a simplified cross-encoder pattern.
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 8):
        np.random.seed(42)
        # Two embeddings concatenated = 2 * input_dim
        self.W1 = np.random.randn(2 * input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.5
        self.b2 = np.zeros(1)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """Score a query-document pair."""
        # Concatenate query and document embeddings
        x = np.concatenate([query_emb, doc_emb])
        # Hidden layer with ReLU
        h = self.relu(x @ self.W1 + self.b1)
        # Output score with sigmoid (0 to 1)
        score = self.sigmoid(h @ self.W2 + self.b2)
        return float(score[0])

    def train(self, examples: list[tuple[str, str, float]], epochs: int = 200, lr: float = 0.1):
        """Train on (query, document, relevance_label) triples."""
        print(f"Training on {len(examples)} examples for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0
            for query, doc, label in examples:
                q_emb = wordVector.embed_text(query, word_vectors)
                d_emb = wordVector.embed_text(doc, word_vectors)

                # Forward pass
                x = np.concatenate([q_emb, d_emb])
                h = self.relu(x @ self.W1 + self.b1)
                score = self.sigmoid(h @ self.W2 + self.b2)

                # Binary cross-entropy loss
                pred = float(score[0])
                loss = -(label * np.log(pred + 1e-8) + (1 - label) * np.log(1 - pred + 1e-8))
                total_loss += loss

                # Backprop (simplified gradient descent)
                d_score = pred - label  # gradient of BCE w.r.t. pre-sigmoid
                d_W2 = h.reshape(-1, 1) * d_score
                d_b2 = np.array([d_score])
                d_h = (self.W2.flatten() * d_score) * (h > 0).astype(float)
                d_W1 = x.reshape(-1, 1) @ d_h.reshape(1, -1)
                d_b1 = d_h

                # Update weights
                self.W2 -= lr * d_W2
                self.b2 -= lr * d_b2
                self.W1 -= lr * d_W1
                self.b1 -= lr * d_h

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: loss = {total_loss / len(examples):.4f}")

    def search(self, query: str, documents: list[str], top_k: int = 3) -> list[tuple[int, float, str]]:
        """Rank documents for a query."""
        q_emb = wordVector.embed_text(query, word_vectors)
        scores = []
        for idx, doc in enumerate(documents):
            d_emb = wordVector.embed_text(doc, word_vectors)
            score = self.forward(q_emb, d_emb)
            scores.append((idx, score, doc))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
# Training data: (query, document, relevance)
# 1.0 = relevant, 0.0 = not relevant
training_data = [
    ("cat", "The cat sat on the mat", 1.0),
    ("cat", "The garden was full of flowers and birds", 0.0),
    ("dog", "Dogs are loyal and friendly pets", 1.0),
    ("dog", "The mat was soft and warm", 0.0),
    ("young animals", "Puppies and kittens are adorable baby animals", 1.0),
    ("young animals", "The garden was full of flowers and birds", 0.0),
    ("pets", "Cats and dogs are the most popular pets worldwide", 1.0),
    ("pets", "The mat was soft and warm", 0.0),
    ("kitten", "A kitten is a young cat", 1.0),
    ("kitten", "The dog chased the cat in the garden", 0.0),
]

ranker = ToyNeuralRanker()
ranker.train(training_data, epochs=120, lr=0.1)
# Test the trained ranker
print("Query: 'cat'")
for idx, score, doc in ranker.search("cat", documents):
    print(f"  [{idx}] score={score:.3f} | {doc}")

print("\nQuery: 'young animals'")
for idx, score, doc in ranker.search("young animals", documents):
    print(f"  [{idx}] score={score:.3f} | {doc}")

print("\nQuery: 'puppy' (unseen in training!)")
for idx, score, doc in ranker.search("puppy", documents):
    print(f"  [{idx}] score={score:.3f} | {doc}")

#Output:
# Training on 10 examples for 120 epochs...
#   Epoch 50: loss = 0.0317
#   Epoch 100: loss = 0.0094
# Query: 'cat'
#   [5] score=1.000 | Puppies and kittens are adorable baby animals
#   [2] score=1.000 | A kitten is a young cat
#   [3] score=1.000 | Dogs are loyal and friendly pets

# Query: 'young animals'
#   [5] score=1.000 | Puppies and kittens are adorable baby animals
#   [2] score=1.000 | A kitten is a young cat
#   [3] score=1.000 | Dogs are loyal and friendly pets

# Query: 'puppy' (unseen in training!)
#   [5] score=0.998 | Puppies and kittens are adorable baby animals
#   [2] score=0.994 | A kitten is a young cat
#   [3] score=0.913 | Dogs are loyal and friendly pets