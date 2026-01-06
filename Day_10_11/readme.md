# Retrieval‑Augmented Generation (RAG) – Full Notes

These notes are written as **professional class notes** for GenAI / LLM courses, with **clear theory + simple Python examples**. Suitable for interviews, exams, and real‑world systems.

---

## 1. What is RAG?

**Retrieval‑Augmented Generation (RAG)** is an LLM architecture where the model:

1. **Retrieves relevant external data** (documents, PDFs, DB records)
2. **Augments the prompt** with retrieved context
3. **Generates an answer grounded in that data**

Instead of relying only on training data, the LLM reasons over **fresh + private knowledge**.

---

## 2. Why RAG is Needed

### Problems with vanilla LLMs

• Hallucinations
• No access to private/company data
• Knowledge cutoff
• Expensive fine‑tuning
• No citations / traceability

### RAG solves

• Grounded responses
• Up‑to‑date information
• Uses internal PDFs, docs, DBs
• Cheaper than fine‑tuning
• Explainable answers

---

## 3. High‑Level RAG Architecture

```
User Query
    ↓
Embedding Model
    ↓
Vector Database (Similarity Search)
    ↓
Top‑K Relevant Chunks
    ↓
Prompt Augmentation
    ↓
LLM
    ↓
Final Answer
```

---

## 4. Core Components of RAG

### 4.1 Document Loader

Loads data from:

• PDFs
• Text files
• Websites
• Databases
• APIs

---

### 4.2 Text Chunking

Documents are split into **small overlapping chunks**.

Why?
• LLM context limits
• Better semantic search
• Higher retrieval accuracy

Typical chunk sizes:
• 300–1000 tokens
• Overlap: 50–200 tokens

---

### 4.3 Embeddings

An **embedding** is a numerical vector representing semantic meaning.

• Similar meaning → vectors closer
• Stored in vector DB

Popular embedding models:
• OpenAI text‑embedding
• Gemini embedding
• BGE
• Instructor
• E5

---

### 4.4 Vector Database

Stores embeddings + metadata and supports similarity search.

Popular vector DBs:
• FAISS (local)
• ChromaDB
• Pinecone
• Weaviate
• Milvus

---

### 4.5 Retriever

Given a query:

• Convert query → embedding
• Perform similarity search
• Return top‑K relevant chunks

---

### 4.6 Prompt Augmentation

Retrieved chunks are injected into the prompt:

```
Answer the question using only the context below:

<context>
...
</context>

Question: ...
```

---

## 5. Simple RAG Flow (End‑to‑End)

1. Load documents
2. Chunk text
3. Create embeddings
4. Store in vector DB
5. Embed user query
6. Retrieve top‑K chunks
7. Send chunks + query to LLM
8. Generate answer

---

## 6. Minimal RAG Example (Pure Python)

### Step 1: Install Dependencies

```bash
pip install faiss-cpu sentence-transformers
```

---

### Step 2: Create Embeddings

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
documents = [
    "RAG combines retrieval and generation",
    "LLMs hallucinate without grounding",
    "Vector databases store embeddings"
]

# Create embeddings
embeddings = model.encode(documents)
```

---

### Step 3: Store in FAISS

```python
# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
```

---

### Step 4: Query Retrieval

```python
query = "Why is RAG useful?"
query_embedding = model.encode([query])

# Search top 2 docs
distances, indices = index.search(query_embedding, k=2)

retrieved_docs = [documents[i] for i in indices[0]]
print(retrieved_docs)
```

---

### Step 5: Prompt Augmentation

```python
context = "\n".join(retrieved_docs)

prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question: {query}
"""

print(prompt)
```

(You now send this prompt to any LLM)

---

## 7. RAG with LLM (Conceptual)

```python
response = llm.generate(prompt)
print(response)
```

LLM answers **only using retrieved knowledge**.

---

## 8. Chunking Strategies

• Fixed size chunking
• Sentence‑based chunking
• Recursive chunking
• Semantic chunking

Best practice:

• Respect paragraph boundaries
• Avoid cutting sentences

---

## 9. RAG vs Fine‑Tuning

| Feature        | RAG     | Fine‑Tuning |
| -------------- | ------- | ----------- |
| Cost           | Low     | High        |
| Updates        | Instant | Retrain     |
| Hallucination  | Low     | Medium      |
| Private Data   | Easy    | Risky       |
| Explainability | High    | Low         |

---

## 10. Advanced RAG Variants

• Hybrid Search (BM25 + vectors)
• Re‑ranking models
• Multi‑query RAG
• Agentic RAG
• Graph‑RAG
• Tool‑calling RAG

---

## 11. Common RAG Mistakes

• Large chunk size
• No overlap
• Low‑quality embeddings
• Too many retrieved chunks
• Prompt not restricting hallucination

---

## 12. Where RAG is Used

• Chat with PDFs
• Internal knowledge bots
• Customer support AI
• Legal / medical QA
• Sales enablement systems
• Agent platforms (like SalesOS / Bronn)

---

## 13. Interview One‑Liners

• "RAG grounds LLM outputs using retrieved external knowledge."
• "It reduces hallucination without retraining the model."
• "Vector similarity search enables semantic retrieval."

---

## 14. Summary

RAG = **Retriever + LLM + Prompt Engineering**

It is the **most production‑ready GenAI architecture today**.

---

If you want:
• RAG with Gemini API
• RAG with PDFs (Flask / FastAPI)
• Agentic RAG
• Evaluation of RAG
• RAG system design for startups

Tell me and I’ll extend this canvas.
