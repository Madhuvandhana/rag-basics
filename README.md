# 📘 Retrieval-Augmented Generation (RAG)

## 🚀 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that improves Large Language Model (LLM) responses by combining:

* 🔍 **Information retrieval** (from external data sources)
* 🧠 **Text generation** (using an LLM)

Instead of relying only on what the model was trained on, RAG allows the model to **fetch relevant, up-to-date, or private data** before generating an answer.

---

## ⚙️ How RAG Works

### 1. 🔎 Retrieve

A user query is converted into a searchable format (often embeddings), and a **vector database** is queried to find relevant documents.

### 2. ➕ Augment

The retrieved documents are added to the prompt along with the original query.

### 3. ✨ Generate

The LLM generates a response using both:

* The user’s question
* The retrieved context

---

## 🧠 Retrieval Strategies

Different strategies can be used to find relevant documents. These fall into **lexical**, **semantic**, and **hybrid** approaches.

---

### 📊 1. Lexical (Keyword-Based) Retrieval

These methods rely on exact or partial word matching.

#### • TF-IDF (Term Frequency–Inverse Document Frequency)

* Scores documents based on importance of words
* Works well for keyword-heavy queries

#### • Jaccard Similarity

* Measures overlap between two sets of words
* Formula:

  ```
  J(A, B) = |A ∩ B| / |A ∪ B|
  ```

#### • Levenshtein Distance

* Measures edit distance between strings
* Useful for:

  * Typos
  * Fuzzy matching
* Lower distance = more similar

---

### 🧬 2. Semantic (Embedding-Based) Retrieval

These methods capture meaning rather than exact wording.

#### • Word Vectors / Embeddings

* Convert text into high-dimensional vectors
* Similar meaning → closer vectors

#### • Cosine Similarity

* Measures angle between vectors
* Formula:

  ```
  similarity = (A · B) / (||A|| ||B||)
  ```
* Most commonly used in vector databases

---

### 🤖 3. Neural Ranking (Re-Rankers)

* Uses deep learning models to rank retrieved documents
* Often applied **after initial retrieval**
* Improves precision by understanding context deeply

Examples:

* Cross-encoders
* Transformer-based rerankers

---

### 🔀 4. Hybrid Retrieval

Combines multiple retrieval methods to get better results.

#### Common Hybrid Setup:

* **BM25 / TF-IDF** (keyword match)
* * **Embedding search** (semantic match)

#### Why Hybrid?

* Lexical → precise keyword matching
* Semantic → meaning understanding
* Together → best of both worlds

---

### ⚖️ 5. Weighted Reciprocal Rank Fusion (RRF)

A method to combine multiple ranked lists from different retrieval systems.

#### Formula:

```
RRF_score = Σ (1 / (k + rank_i))
```

* `rank_i`: rank from each retrieval method
* `k`: smoothing constant (usually ~60)

#### Benefits:

* Combines multiple strategies robustly
* Avoids reliance on a single retrieval method
* Simple but very effective

---

## 🧱 Typical RAG Pipeline

```
User Query
     ↓
Embedding / Tokenization
     ↓
Retrieval (Vector DB / Hybrid Search)
     ↓
(Optional) Re-ranking
     ↓
Context Augmentation
     ↓
LLM Generation
     ↓
Final Answer
```

---

## 🛠️ Key Components

* **Vector Database** (e.g., FAISS, Pinecone, Weaviate)
* **Embedding Model** (e.g., OpenAI, sentence-transformers)
* **Retriever** (lexical / semantic / hybrid)
* **LLM** (for generation)
* **Re-ranker** (optional but powerful)

---

## ✅ When to Use RAG

* Private/internal data (docs, PDFs, logs)
* Frequently changing information
* Reducing hallucinations
* Building AI assistants over custom knowledge

---

## ⚡ Summary

| Approach        | Strength                 | Weakness                  |
| --------------- | ------------------------ | ------------------------- |
| TF-IDF / BM25   | Exact keyword match      | Misses semantic meaning   |
| Jaccard         | Simple overlap           | Ignores importance        |
| Levenshtein     | Handles typos            | Not semantic              |
| Cosine (Embeds) | Captures meaning         | Needs good embeddings     |
| Neural Rankers  | High accuracy            | Expensive                 |
| Hybrid          | Best overall performance | More complex              |
| RRF             | Robust ranking fusion    | Needs multiple retrievers |

---

