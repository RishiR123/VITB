## Day 6 – Transformers & Attention (Theory + Practicals)

### Overview

Transformers are the backbone of modern AI models like GPT, BERT, Gemini, and LLaMA. They replaced older sequence models by using **attention** instead of recurrence, allowing models to understand context, meaning, and relationships in text efficiently.

---

## PART 1: THEORY

### Why Do We Need Transformers?

Earlier models like RNNs and LSTMs:

* Process text **word by word**
* Are **slow** for long sequences
* Struggle to remember long-range context

Transformers solve this by:

* Processing **all words at once**
* Using **attention** to focus on important words
* Scaling efficiently on GPUs

---

### What is Attention?

Attention tells the model:

> Which words should I focus on while understanding this word?

Example:
"The animal didn’t cross the road because it was tired."

Attention helps the model understand that **“it” refers to “animal”**, not the road.

---

### Self-Attention (Simple Explanation)

Each word looks at every other word in the sentence and decides how important they are.

Each word creates:

* **Query (Q)** – what am I looking for?
* **Key (K)** – what do I contain?
* **Value (V)** – what information do I provide?

The model compares queries with keys to decide which values to focus on.

---

### Multi-Head Attention

Instead of one attention mechanism, transformers use multiple attention heads.

Each head learns different relationships:

* Grammar
* Meaning
* Subject–object relations

This improves understanding significantly.

---

### Transformer Architecture (High Level)

A transformer block consists of:

1. Embedding Layer
2. Positional Encoding
3. Multi-Head Self-Attention
4. Feed Forward Neural Network
5. Residual Connections + Layer Normalization

Multiple such blocks are stacked together.

---

### Why Positional Encoding?

Transformers process words in parallel, so they do not inherently know word order.

Positional encoding adds position information to embeddings so the model understands sequence order.

---

### Encoder vs Decoder

* **Encoder-only**: BERT (understanding tasks)
* **Decoder-only**: GPT (text generation)
* **Encoder–Decoder**: T5 (translation)

---

### Why Transformers Are Powerful

* Capture long-range dependencies
* Parallel processing (fast)
* Scalable
* Foundation of LLMs, RAG, and Agentic systems

---

## PART 2: PRACTICALS

### Practical 1: Simple Attention Concept

```python
import numpy as np

query = np.array([1, 0, 1])
keys = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0]
])

scores = np.dot(keys, query)
print(scores)
```

Higher scores indicate higher attention.

---

### Practical 2: Transformer-Based Text Generation

```python
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain transformers in simple terms"
)

print(response.text)
```

This demonstrates how transformer models generate context-aware responses.

---

### Key Takeaways

* Attention is the core idea behind transformers
* Transformers process text in parallel
* All modern LLMs are transformer-based

---

## RNN – Recurrent Neural Networks (Simplified)

### Overview

RNNs were designed to process sequential data by maintaining a memory of past inputs.

---

## PART 1: THEORY

### What is an RNN?

An RNN processes data one step at a time and passes information forward using a **hidden state**.

Example sentence:
"I love AI"

The model processes:

* I → love → AI

Each step depends on the previous step.

---

### Hidden State

* Stores information from previous steps
* Acts as short-term memory
* Updated at every time step

---

### Problems with RNNs

* Vanishing gradient problem
* Difficulty remembering long-term dependencies
* Slow due to sequential processing

---

### LSTM and GRU

To overcome RNN limitations:

* **LSTM** uses memory gates
* **GRU** is a simplified LSTM

Despite improvements, they are slower than transformers.

---

### RNN vs Transformer

| Feature      | RNN        | Transformer |
| ------------ | ---------- | ----------- |
| Processing   | Sequential | Parallel    |
| Memory       | Limited    | Long-range  |
| Speed        | Slow       | Fast        |
| Modern Usage | Rare       | Very Common |

---

## PART 2: PRACTICAL

### Simple RNN Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential()
model.add(SimpleRNN(32, input_shape=(10, 1)))
model.add(Dense(1))

model.summary()
```

This shows how RNNs process sequences step by step.

---

### Final Key Takeaways

* RNNs introduced sequence learning
* Transformers solved RNN limitations
* Modern AI systems rely on attention mechanisms

---

### What Comes Next

**Day 7: LLM Evaluation & Hallucinations**

Students must clearly understand why transformers replaced RNNs before studying LLM evaluation and safety.
