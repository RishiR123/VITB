# Day 5 – Embeddings

## Assignment: Text Embeddings & Similarity

### Objective

This assignment helps students understand how **text embeddings** work and how they are used to measure **semantic similarity** between pieces of text. Students will generate embeddings, compare vectors, and simulate a basic semantic search.

---

## Instructions

* Use **Python 3**
* Use **Google GenAI embeddings API**
* Use `numpy` for similarity calculation
* Write clear, commented code
* Save the file as `day5_embeddings.py`

---

## Task 1: Generate an Embedding

1. Create an embedding for the following sentence:

```
"Artificial Intelligence is shaping the future"
```

2. Print:

* The length of the embedding vector
* The first 5 values of the vector

Expected Outcome:

* A fixed-size numerical vector

---

## Task 2: Embedding Similarity

1. Generate embeddings for the following two sentences:

```
Sentence A: "AI is transforming industries"
Sentence B: "Artificial intelligence is changing businesses"
```

2. Compute **cosine similarity** between the two embeddings.
3. Print the similarity score.

Expected Outcome:

* A similarity score closer to 1

---

## Task 3: Similar vs Unrelated Text

1. Generate embeddings for:

```
Text 1: "I love machine learning"
Text 2: "Deep learning models are powerful"
Text 3: "The weather is very hot today"
```

2. Compute similarity:

* Text 1 vs Text 2
* Text 1 vs Text 3

3. Compare the scores.

Expected Outcome:

* Higher similarity for related texts

---

## Task 4: Mini Semantic Search

1. Create a list of sentences:

```
[
  "I enjoy studying AI",
  "Football is a popular sport",
  "Machine learning models learn from data",
  "The sky is very blue today"
]
```

2. Use the query:

```
"learning artificial intelligence"
```

3. Generate embeddings for the query and all sentences.
4. Rank the sentences based on cosine similarity.
5. Print the sentences in descending order of relevance.

Expected Outcome:

* AI-related sentences should rank highest

---

## Bonus Task (Optional)

* Change the query text and observe ranking changes
* Add more sentences and test robustness

---

## Submission Guidelines

* File name: `day5_embeddings.py`
* Code must run without errors
* Include printed outputs

---

## Learning Outcome

After completing this assignment, students should be able to:

* Explain what embeddings represent
* Generate embeddings using an API
* Measure semantic similarity
* Build a simple semantic search system

---

## Deadline

Submit before the start of **Day 6 session**.
