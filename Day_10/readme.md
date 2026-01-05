# 📘 LLM Inference – Class Notes

## 1️⃣ What is LLM Inference?

**LLM Inference** is the process of using a **trained Large Language Model** to generate outputs (text, answers, summaries) from a given input prompt.

> Training = learning weights ❌
> Inference = using learned weights to generate output ✅

---

## 2️⃣ Inference Pipeline (High Level)

```
Input Text → Tokenizer → Model → Generated Tokens → Decoder → Output Text
```

### Components:

* **Tokenizer**: Converts text → numbers (tokens)
* **Model**: Predicts next tokens
* **Decoder**: Converts tokens → readable text

---

## 3️⃣ Popular Packages for LLM Inference

| Package        | Use Case                   |
| -------------- | -------------------------- |
| `transformers` | Most common (Hugging Face) |
| `torch`        | Model execution backend    |
| `accelerate`   | Multi‑GPU / optimization   |
| `vllm`         | High‑speed inference       |

---

## 4️⃣ Installing Required Packages

```bash
pip install transformers torch
```

---

## 5️⃣ Simple LLM Inference (Step‑by‑Step)

### ✅ Example Model

We use **GPT‑2** because:

* Small size
* Works on CPU
* No authentication required

---

### 📌 Code: Basic LLM Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input prompt
prompt = "Artificial Intelligence is"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True
)

# Decode tokens to text
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

---

## 6️⃣ Explanation of Important Parameters

| Parameter        | Meaning                                         |
| ---------------- | ----------------------------------------------- |
| `max_new_tokens` | Max tokens to generate                          |
| `temperature`    | Creativity control (0.2 = safe, 1.0 = creative) |
| `do_sample`      | Enables randomness                              |

---

## 7️⃣ Simplified Inference using Pipeline API

Best for **quick demos & teaching**.

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

output = generator(
    "Artificial Intelligence is",
    max_new_tokens=50
)

print(output[0]["generated_text"])
```

---

## 8️⃣ CPU vs GPU Inference

### CPU (Default)

```python
device = -1
```

### GPU (CUDA)

```python
device = 0
```

```python
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0
)
```

---

## 9️⃣ Real‑World Use Cases of LLM Inference

* Chatbots
* Resume parsing (ATS)
* Code generation
* Question answering
* RAG systems
* AI Agents

---

## 🔟 Key Takeaways

* Inference ≠ Training
* Tokenizer + Model + Decoder = LLM
* `transformers` is the industry standard
* Pipeline API is best for beginners

---

## 🧪 Assignment (Optional)

1. Change the prompt and observe output
2. Modify `temperature` values
3. Try another model (e.g., `distilgpt2`)

---

## 🔜 Next Class Topics

* Gemini API inference
* LLaMA local inference
* FastAPI inference server
* LLM Evaluation basics

---

✅ *End of Notes*
