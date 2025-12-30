# Day 5 – Embeddings (Theory + Practicals)

## Overview

Embeddings are numerical representations of text that capture **semantic meaning**. Similar texts have embeddings that are close to each other in vector space. Embeddings are the foundation for **search, recommendation systems, clustering, and RAG pipelines**.

---

## PART 1: THEORY

### What are Embeddings?

* Embeddings convert text into a list of numbers (vectors)
* These vectors represent meaning, not exact words
* Words or sentences with similar meaning have similar vectors

Example:

* "AI is powerful" ≈ "Artificial intelligence is strong"
* Their embeddings will be close in vector space

---

### Why Embeddings Matter

Embeddings are used for:

* Semantic search
* Question answering
* Recommendation systems
* Document similarity
* Retrieval-Augmented Generation (RAG)

Without embeddings, AI systems rely only on keyword matching.

---

### Vector Concepts (Simple)

* A vector is a list of numbers: `[0.12, -0.45, 0.88, ...]`
* Each number represents a learned feature
* Embedding models produce vectors with fixed size (e.g., 384, 768, 1536)

---

### Similarity Between Vectors

The most common similarity measure is **Cosine Similarity**.

* Output range: `-1 to 1`
* `1` → identical meaning
* `0` → unrelated
* `-1` → opposite meaning

---

## PART 2: PRACTICALS

### Practical 1: Create Text Embeddings using Google GenAI

#### Step 1: Setup

```python
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")
```

---

#### Step 2: Generate Embeddings

```python
response = client.models.embed_content(
    model="text-embedding-004",
    content="NLP is a core part of artificial intelligence"
)

embedding = response
print(embedding)
```

Explanation:

* The output is a vector (list of numbers)
* Vector length depends on the model

---

### Practical 2: Compare Similarity Between Two Sentences

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

text1 = "AI is transforming the world"
text2 = "Artificial intelligence is changing industries"

emb1 = client.models.embed_content(
    model="text-embedding-004",
    content=text1
)['embedding']

emb2 = client.models.embed_content(
    model="text-embedding-004",
    content=text2
)['embedding']

similarity = cosine_similarity(emb1, emb2)
print("Similarity:", similarity)
```

Expected Observation:

* Similar meaning → high similarity score

---

### Practical 3: Embedding-Based Search (Mini Demo)

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

sentences = [
    "I love machine learning",
    "The weather is very hot today",
    "AI models use neural networks",
    "I enjoy studying artificial intelligence"
]

query = "learning AI"

# Embed query
query_response = client.models.embed_content(
    model="text-embedding-004",
    contents=query
)
query_emb = np.array(query_response.embeddings[0].values)

scores = []

for sent in sentences:
    sent_response = client.models.embed_content(
        model="text-embedding-004",
        contents=sent
    )
    sent_emb = np.array(sent_response.embeddings[0].values)

    score = cosine_similarity(query_emb, sent_emb)
    scores.append((sent, score))

# Sort by similarity
scores.sort(key=lambda x: x[1], reverse=True)

for s, sc in scores:
    print(s, "->", sc)

```

Explanation:

* This simulates semantic search
* Top results are meaning-based, not keyword-based

---

## Key Takeaways

* Embeddings represent meaning as vectors
* Similar text → closer vectors
* Cosine similarity measures semantic closeness
* Embeddings are the backbone of RAG and agent memory systems

---

## What Comes Next

* Day 6: Transformers and Attention
* Day 7: LLM Evaluation and Hallucinations

Students should be comfortable with embeddings before moving into RAG pipelines.
