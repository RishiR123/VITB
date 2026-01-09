# RAG Curriculum – Days 13 to 17 (Professional Notes)

---

## Day 13 – Chunking, Loaders & Splitters (Foundation of RAG)

### 1. Why This Matters

In RAG systems, **retrieval quality depends directly on chunking quality**. Poor chunking leads to irrelevant retrieval and hallucinations, even with strong LLMs.

---

### 2. Document Loaders

**Definition:** Loaders ingest raw data and convert it into structured text with metadata.

**Common Data Sources:**

* PDF files
* Word documents
* HTML / Web pages
* Markdown files
* Databases
* APIs

**Typical Output Structure:**

* `page_content`: extracted text
* `metadata`: source, page number, URL, section, etc.

Metadata is essential for **filtering, routing, and citation**.

---

### 3. Chunking

**Why chunking is required:**

* LLMs have limited context windows
* Embeddings perform better on focused semantic units

---

### 4. Chunking Strategies

#### a) Fixed-size Chunking

* Splits text into equal-sized chunks (e.g., 500 tokens)
* Simple and fast
* May break semantic meaning

#### b) Overlapping Chunking

* Adjacent chunks overlap by 10–20%
* Preserves context across boundaries

#### c) Recursive Chunking (Recommended Default)

* Splits hierarchically: paragraph → sentence → word
* Works well for PDFs and technical documents

#### d) Semantic Chunking

* Uses embeddings to split based on meaning
* High quality but computationally expensive

---

### 5. Chunking Best Practices

* Chunk size: **300–800 tokens**
* Overlap: **10–20%**
* Always chunk before embedding
* Manually test retrieval quality

---

### 6. Splitters

Splitters define *how* text is broken:

* Character-based
* Token-based
* Sentence-based
* Recursive splitters

> Interview insight: Most RAG failures come from poor chunking, not poor models.

---

## Day 14 – RAG Pipeline: Retriever + LLM

### 1. End-to-End RAG Pipeline

Query → Embedding → Retriever → Top-K Chunks → Prompt → LLM → Answer

---

### 2. Retriever

**Role:** Finds the most relevant chunks for a given query.

**Similarity Methods:**

* Cosine similarity
* Dot product
* Hybrid (BM25 + vectors)

**Top-K Selection Guidelines:**

* K = 3–5 → precision-focused
* K = 6–8 → recall-focused
* Too large K introduces noise

---

### 3. Prompt Construction

**Weak Prompt:**
"Answer using the context"

**Strong Prompt:**

* Enforce factual grounding
* Explicitly restrict answers to retrieved context
* Define behavior when answer is missing

---

### 4. Role of the LLM

* Does not search
* Does not store external knowledge
* Only reasons over retrieved content

---

### 5. Common Failure Modes

* Irrelevant retrieval
* Excessive context
* Vague prompts

---

## Day 15 – RAG Evaluation: Relevance & Grounding

### 1. Why Evaluation Is Mandatory

RAG systems may appear correct while silently hallucinating. Evaluation ensures **trust and reliability**.

---

### 2. Core Evaluation Dimensions

#### a) Retrieval Relevance

* Did the retriever fetch the correct chunks?
* Metrics: Recall@K, Precision@K

#### b) Answer Relevance

* Does the answer address the user question?

#### c) Grounding / Faithfulness (Most Critical)

* Is the answer strictly supported by retrieved context?

---

### 3. Hallucination Types

* Unsupported facts
* Over-generalization
* Mixing external knowledge

---

### 4. Evaluation Approaches

* Rule-based checks
* LLM-as-a-Judge evaluation

---

### 5. Evaluation Tools

* RAGAS
* TruLens
* OpenEval
* Custom LLM evaluators

> Production rule: No RAG system should go live without automated evaluation.

---

## Day 16 – Advanced RAG: Query Routing

### 1. Why Query Routing

Different queries require different retrieval strategies. Routing improves **accuracy, speed, and cost**.

---

### 2. Query Types

* Factual
* Analytical
* Navigational
* Conversational (follow-ups)

---

### 3. Query Routing Architecture

User Query → Router (Rule / LLM) → Data Source → Retrieval / Tool

---

### 4. Routing Strategies

#### a) Rule-based Routing

* Keyword or pattern matching
* Simple and deterministic

#### b) LLM-based Routing

* LLM decides which data source or tool to use
* Can route to vector DB, SQL, web search, or no retrieval

---

### 5. Benefits of Routing

* Reduced latency
* Lower cost
* Higher relevance

---

## Day 17 – Context Optimization & Compression

### 1. The Problem

* Limited LLM context windows
* Retrieved chunks often contain redundant or irrelevant information

---

### 2. Context Compression

**Definition:** Reducing retrieved context while preserving meaning and facts.

---

### 3. Compression Techniques

#### a) Extractive Compression

* Remove irrelevant sentences
* Keep key facts

#### b) Abstractive Compression

* LLM-generated summaries
* Compact but higher risk of information loss

#### c) Redundancy Removal

* Eliminate duplicate or overlapping content

---

### 4. Context Ranking

Reorder chunks based on:

* Relevance score
* Alignment with the question

---

### 5. Optimized RAG Flow

Retrieve → Rank → Compress → Inject → Generate

> Key insight: Good RAG is about the **right context**, not more context.

---

## Overall Learning Outcome

By Day 17, learners can design, evaluate, and optimize **production-grade RAG systems** with confidence.
