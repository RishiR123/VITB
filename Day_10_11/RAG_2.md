# RAG Curriculum – Days 13 to 17 (Professional Notes)

---

## Day 13 – Chunking, Loaders & Splitters (Foundation of RAG)

### Practical 13.1 – Load Documents (PDF/Text)

**Explanation:**
This step ingests raw documents into the RAG system. A loader reads files (PDF, text, etc.) and converts them into a standardized `Document` object containing:

* `page_content`: the extracted text
* `metadata`: page number, source file, URL, etc.

Loaders are the **entry point of knowledge** into RAG. If text extraction is wrong or incomplete, retrieval will fail regardless of model quality.

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")
documents = loader.load()
print(len(documents))
print(documents[0].page_content[:300])
```

---

### Practical 13.2 – Chunking with Recursive Splitter

**Explanation:**
LLMs and embedding models work best on focused semantic units. Recursive chunking splits documents hierarchically (paragraph → sentence → word) until chunks fit the desired size.

Why this matters:

* Prevents breaking sentences unnaturally
* Preserves meaning better than fixed splitting
* Works well for PDFs and technical documents

Chunk size and overlap directly control retrieval accuracy.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)
print(len(chunks))
```

---

### Practical 13.3 – Store Chunks in Vector DB (FAISS)

**Explanation:**
Each chunk is converted into an embedding (vector representation of meaning). These vectors are stored in a vector database (FAISS) to enable similarity search.

Why FAISS:

* Fast
* Open-source
* Ideal for local experimentation

At this stage, the system has a **searchable knowledge base**.

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
```

> Practical takeaway: Chunk size and overlap directly affect retrieval quality.

> Interview insight: Most RAG failures come from poor chunking, not poor models.

---

## Day 14 – RAG Pipeline: Retriever + LLM

### Practical 14.1 – Build Retriever

**Explanation:**
The retriever is responsible for searching the vector database and returning the most relevant chunks for a user query.

Key idea:

* The retriever does **semantic search**, not keyword search
* It compares query embeddings with stored chunk embeddings

The `k` value controls how many chunks are returned.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

---

### Practical 14.2 – Query Retrieval

**Explanation:**
Here, a user question is converted into an embedding and matched against stored vectors. The output is a list of the most relevant document chunks.

This step answers:

* *Did the system find the right information?*

Always manually inspect retrieved chunks during development.

```python
query = "What is Retrieval Augmented Generation?"
retrieved_docs = retriever.get_relevant_documents(query)

for doc in retrieved_docs:
    print(doc.page_content[:200])
```

---

### Practical 14.3 – RAG Prompt + LLM Call

**Explanation:**
This is where generation happens. Retrieved chunks are injected into a carefully designed prompt and passed to the LLM.

Important rules:

* LLM must answer **only from context**
* If context is missing, the model should explicitly say so

The LLM does not retrieve data—it only reasons over provided text.

```python
from langchain.llms import HuggingFacePipeline

prompt = f"""
Answer strictly using the context below.
If the answer is not present, say 'Not in context'.

Context:
{retrieved_docs}

Question:
{query}
"""

llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation"
)

response = llm(prompt)
print(response)
```

---

### Common Failure Modes

* Irrelevant retrieval
* Excessive context
* Vague prompts

---

## Day 15 – RAG Evaluation: Relevance & Grounding

### Practical 15.1 – Retrieval Relevance Check

**Explanation:**
This evaluates whether the retriever fetched useful chunks.

Precision measures:

* How many retrieved chunks are actually relevant

Poor retrieval relevance means the generator will fail even with a perfect prompt.

```python
relevant = sum([1 for d in retrieved_docs if "RAG" in d.page_content])
precision = relevant / len(retrieved_docs)
print("Precision:", precision)
```

---

### Practical 15.2 – LLM as Judge (Faithfulness)

**Explanation:**
Faithfulness checks whether the generated answer is grounded in the retrieved context.

Using an LLM as a judge allows:

* Scalable evaluation
* Detection of hallucinations

This step is critical for production-grade RAG systems.

```python
eval_prompt = f"""
Check if the answer is supported by the context.
Answer Yes or No.

Context:
{retrieved_docs}

Answer:
{response}
"""

eval_result = llm(eval_prompt)
print("Grounded:", eval_result)
```

---

> Production rule: No RAG system should go live without automated evaluation.

---

## Day 16 – Advanced RAG: Query Routing

### Practical 16.1 – Rule-Based Query Routing

**Explanation:**
Not all queries need the same retrieval depth. Rule-based routing uses simple heuristics to decide how a query should be handled.

Examples:

* Definitions → small index
* Comparisons → deeper retrieval

This improves speed and reduces cost.

```python
def route_query(query):
    if "compare" in query.lower():
        return "deep_retrieval"
    elif "define" in query.lower():
        return "faq"
    return "general"

route = route_query(query)
print("Route:", route)
```

---

### Practical 16.2 – LLM-Based Router

**Explanation:**
Instead of hard-coded rules, an LLM can classify queries dynamically.

Advantages:

* Handles complex language
* Adapts to new query patterns

The router decides whether to:

* Retrieve documents
* Use a database
* Answer directly

```python
router_prompt = f"""
Classify the query into one of these:
faq, deep_retrieval, no_retrieval

Query:
{query}
"""

route = llm(router_prompt)
print(route)
```

---

### Benefits of Routing

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

---

# Detailed Explanation of All Practicals (Step-by-Step)

This section explains **what each practical does, why it exists, and how it fits into the RAG system**. This is meant for deep understanding, not just execution.

---

## Day 13 Practicals – Chunking, Loaders & Vector Storage

### Practical 13.1 – Document Loading

**What happens:**

* A loader reads a PDF and extracts text page by page.
* Each page becomes a `Document` object.

**Why this is needed:**
LLMs cannot read raw files. All external knowledge must be converted into clean text first.

**Key concept learned:**
Data ingestion layer of RAG.

---

### Practical 13.2 – Recursive Chunking

**What happens:**

* Large documents are split into smaller chunks.
* Recursive splitting preserves semantic boundaries (paragraphs, sentences).

**Why this is needed:**

* Embeddings work best on focused text
* Prevents context overflow

**Key concept learned:**
Chunk size and overlap directly control retrieval accuracy.

---

### Practical 13.3 – Vector Store Creation (FAISS)

**What happens:**

* Each chunk is converted into an embedding (vector)
* Vectors are stored in FAISS for similarity search

**Why this is needed:**
Traditional databases cannot search by meaning. Vector databases enable semantic search.

**Key concept learned:**
Knowledge storage layer of RAG.

---

## Day 14 Practicals – Retriever + LLM (Core RAG Pipeline)

### Practical 14.1 – Retriever Creation

**What happens:**

* The vector database is wrapped as a retriever
* Top-K similar chunks are fetched per query

**Why this is needed:**
LLMs should only see relevant knowledge, not the entire dataset.

---

### Practical 14.2 – Query Retrieval

**What happens:**

* User query is embedded
* Similar chunks are retrieved using cosine similarity

**Why this is needed:**
This is where factual grounding starts. Bad retrieval = bad answers.

---

### Practical 14.3 – RAG Prompt + Generation

**What happens:**

* Retrieved chunks are injected into the prompt
* LLM is forced to answer only from context

**Why this is needed:**
Prevents hallucinations and ensures explainable answers.

**Key concept learned:**
LLM is a reasoning engine, not a knowledge store.

---

## Day 15 Practicals – Evaluation (Trust & Safety)

### Practical 15.1 – Retrieval Relevance (Precision)

**What happens:**

* Checks how many retrieved chunks are actually relevant

**Why this is needed:**
Even strong LLMs fail if retriever quality is low.

---

### Practical 15.2 – Grounding / Faithfulness Check

**What happens:**

* LLM judges whether the answer is supported by context

**Why this is needed:**
RAG systems often hallucinate subtly. This catches silent failures.

**Key concept learned:**
Evaluation is mandatory for production RAG.

---

## Day 16 Practicals – Query Routing (Advanced Intelligence)

### Practical 16.1 – Rule-Based Routing

**What happens:**

* Simple logic routes queries to different retrieval paths

**Why this is needed:**
Not all questions require deep retrieval.

---

### Practical 16.2 – LLM-Based Routing

**What happens:**

* LLM decides whether retrieval is required
* Routes to FAQ, deep search, or no retrieval

**Why this is needed:**
Improves speed, cost efficiency, and accuracy.

**Key concept learned:**
RAG becomes adaptive and intelligent.

---

## Day 17 Concepts – Context Optimization (Conceptual Explanation)

Although implementation varies, the idea is consistent:

**Retrieve → Rank → Compress → Generate**

**Why this matters:**

* LLM context is expensive
* Less but relevant context gives better answers

**Final insight:**
Good RAG systems win not by bigger models, but by better context control.
