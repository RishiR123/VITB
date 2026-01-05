# 📘 LLM Evaluation – Theory & Practical (GenAI Course)

---

## 1️⃣ What is LLM Evaluation?

**LLM Evaluation** is the process of measuring how well a Large Language Model performs on a given task in terms of:

* Correctness
* Clarity
* Relevance
* Instruction following
* Safety

Unlike traditional ML models, LLM outputs are **open‑ended**, so evaluation is not based only on accuracy.

---

## 2️⃣ Why LLM Evaluation is Needed

LLMs may:

* Hallucinate facts
* Give fluent but incorrect answers
* Respond inconsistently to similar prompts
* Fail silently in real‑world applications

👉 Therefore, **evaluation is continuous and multi‑dimensional**.

---

## 3️⃣ Traditional ML vs LLM Evaluation

| Traditional ML        | LLMs                   |
| --------------------- | ---------------------- |
| Fixed output          | Open‑ended text        |
| Single correct answer | Multiple valid answers |
| Accuracy-based        | Quality-based          |
| Fully automated       | Needs judgment         |

---

## 4️⃣ Types of LLM Evaluation

### 4.1 Automatic / Metric‑Based Evaluation

Used when a reference answer exists.

**Metrics:**

* Accuracy
* Precision / Recall / F1
* Exact Match
* BLEU / ROUGE

**Use cases:**

* Classification
* Named Entity Recognition
* QA with known answers

---

### 4.2 Human Evaluation

Humans score responses based on:

* Correctness
* Clarity
* Relevance
* Tone

❌ Expensive
❌ Time‑consuming
❌ Subjective

---

### 4.3 LLM‑as‑a‑Judge (Most Used in Industry)

A strong LLM evaluates the output of another LLM using a **rubric‑based prompt**.

✅ Scalable
✅ Cost‑effective
✅ Suitable for reasoning tasks

⚠️ Judge bias may exist

---

## 5️⃣ Evaluation Dimensions

1. Correctness
2. Instruction following
3. Clarity & coherence
4. Factual accuracy
5. Hallucination rate
6. Safety & toxicity

---

## 6️⃣ Offline vs Online Evaluation

### Offline Evaluation

* Done before deployment
* Fixed test dataset
* Used for benchmarking

### Online Evaluation

* Done after deployment
* User feedback
* Success/failure logs

---

## 7️⃣ Popular LLM Evaluation Packages

| Package               | Purpose                    |
| --------------------- | -------------------------- |
| OpenEvals             | LLM‑as‑a‑judge evaluation  |
| DeepEval              | Test‑driven LLM evaluation |
| lm‑evaluation‑harness | Benchmarking LLMs          |
| HuggingFace evaluate  | BLEU, ROUGE, BERTScore     |
| LangSmith / Langfuse  | Observability + evaluation |

---

# 🧪 PRACTICAL: LLM Evaluation using Gemini API

## 🎯 Aim

To generate text using Gemini and evaluate it using **LLM‑as‑a‑Judge**.

---

## 🛠 Requirements

* Python 3.9+
* Gemini API Key
* Internet access

### Install Dependency

```bash
pip install google-generativeai
```

---

## 🔑 Step 1: Gemini API Setup

```python
from google import genai

genai.Client(api_key="YOUR_GEMINI_API_KEY")
```

---

## 🧠 Step 2: Generate Response (Model Under Test)

```python
model = genai.GenerativeModel("gemini-pro")

prompt = "Explain Transformers in simple words for beginners."
response = model.generate_content(prompt)
model_answer = response.text

print(model_answer)
```

---

## 🧪 Step 3: Create Evaluation Prompt (LLM‑as‑a‑Judge)

```python
eval_prompt = f"""
You are an evaluator.

Evaluate the following answer based on:
1. Correctness
2. Clarity
3. Simplicity

Give scores from 1 to 5 for each.
Return output strictly in JSON.

Answer:
"""{model_answer}"""
"""
```

---

## 📊 Step 4: Run Evaluation

```python
judge_response = model.generate_content(eval_prompt)
print(judge_response.text)
```

### Sample Output

```json
{
  "correctness": 4,
  "clarity": 5,
  "simplicity": 4,
  "comment": "Clear and beginner‑friendly explanation"
}
```

---

## 📈 Step 5: Score Aggregation

```python
import json
scores = json.loads(judge_response.text)

average = (
    scores["correctness"] +
    scores["clarity"] +
    scores["simplicity"]
) / 3

print("Average Score:", average)
```

---

## 🔁 Step 6: Prompt Comparison (Optional)

```python
prompt1 = "Explain Transformers"
prompt2 = "Explain Transformers using a real‑life analogy"

resp1 = model.generate_content(prompt1).text
resp2 = model.generate_content(prompt2).text
```

Evaluate both responses and compare scores.

---

## 📝 Student Assignment

1. Generate answers for 3 prompts using Gemini
2. Evaluate each using LLM‑as‑a‑Judge
3. Compare scores
4. Write conclusions

---

## ✅ Key Takeaways

* LLM evaluation is not accuracy‑based
* Multiple answers can be correct
* LLM‑as‑a‑Judge is widely used
* Evaluation is continuous in real systems

---

---

# 🧪 PRACTICAL 2: LLM Evaluation using **OpenEvals + Gemini API**

This experiment demonstrates **package-based LLM evaluation** using **OpenEvals** with **Gemini as the Judge Model**.

---

## 🛠 Requirements

* Python 3.9+
* Gemini API Key

### Install Dependencies

```bash
pip install google-generativeai openevals
```

---

## 🔑 Step 1: Configure Gemini API

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

---

## 🧠 Step 2: Generate Output (Model Under Test)

```python
model = genai.GenerativeModel("gemini-pro")

prompt = "Explain Transformers in simple words for a beginner."
response = model.generate_content(prompt)
model_output = response.text

print("MODEL OUTPUT:
", model_output)
```

---

## ⚖️ Step 3: LLM-as-a-Judge using OpenEvals

OpenEvals provides **predefined evaluation rubrics**.

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
```

---

## 🧪 Step 4: Create Evaluator (Gemini as Judge)

```python
judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="google:gemini-pro"
)
```

---

## 📊 Step 5: Run Evaluation

```python
result = judge(
    inputs=prompt,
    outputs=model_output
)

print("EVALUATION RESULT:
", result)
```

### Sample Output

```json
{
  "score": 0.8,
  "reason": "The explanation is mostly correct and suitable for beginners."
}
```

---

## 📈 Step 6: Interpret Results

* Score range: **0 to 1**
* Higher score → better correctness
* Reason explains evaluation decision

---

## 🔁 Step 7: Compare Two Prompts (Recommended)

```python
prompt_a = "Explain Transformers"
prompt_b = "Explain Transformers with a real-life analogy"

out_a = model.generate_content(prompt_a).text
out_b = model.generate_content(prompt_b).text

res_a = judge(inputs=prompt_a, outputs=out_a)
res_b = judge(inputs=prompt_b, outputs=out_b)

print("Prompt A Score:", res_a)
print("Prompt B Score:", res_b)
```

---

## 📝 Student Assignment (OpenEvals)

1. Generate outputs for 2 prompts
2. Evaluate using OpenEvals
3. Compare scores
4. Write observations

---

## ✅ Learning Outcome

* Use of **evaluation package** instead of manual prompts
* Understanding of **LLM-as-a-Judge**
* Comparison of prompt quality
* Industry-style evaluation workflow

---

📌 **End of Experiment**
