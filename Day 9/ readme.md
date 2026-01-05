📘 LLM Evaluation – Theory & Practical (GenAI Course)


---

1️⃣ What is LLM Evaluation?

LLM Evaluation is the process of measuring how well a Large Language Model performs on a given task in terms of:

Correctness

Clarity

Relevance

Instruction following

Safety


Unlike traditional ML models, LLM outputs are open-ended, so evaluation is not based only on accuracy.


---

2️⃣ Why LLM Evaluation is Needed

LLMs may:

Hallucinate facts

Give fluent but incorrect answers

Respond inconsistently to similar prompts

Fail silently in real-world applications


👉 Therefore, evaluation is continuous and multi-dimensional.


---

3️⃣ Traditional ML vs LLM Evaluation

Traditional ML	LLMs

Fixed output	Open-ended text
Single correct answer	Multiple valid answers
Accuracy-based	Quality-based
Fully automated	Needs judgment



---

4️⃣ Types of LLM Evaluation

4.1 Automatic / Metric-Based Evaluation

Used when a reference answer exists.

Metrics:

Accuracy

Precision / Recall / F1

Exact Match

BLEU / ROUGE


Use cases:

Classification

Named Entity Recognition

QA with known answers



---

4.2 Human Evaluation

Humans score responses based on:

Correctness

Clarity

Relevance

Tone


❌ Expensive ❌ Time-consuming ❌ Subjective


---

4.3 LLM-as-a-Judge (Most Used in Industry)

A strong LLM evaluates the output of another LLM using a rubric-based prompt.

✅ Scalable ✅ Cost-effective ✅ Suitable for reasoning tasks

⚠️ Judge bias may exist


---

5️⃣ Evaluation Dimensions

1. Correctness


2. Instruction following


3. Clarity & coherence


4. Factual accuracy


5. Hallucination rate


6. Safety & toxicity




---

6️⃣ Offline vs Online Evaluation

Offline Evaluation

Done before deployment

Fixed test dataset

Used for benchmarking


Online Evaluation

Done after deployment

User feedback

Success/failure logs



---

7️⃣ Popular LLM Evaluation Packages

Package	Purpose

OpenEvals	LLM-as-a-judge evaluation
DeepEval	Test-driven LLM evaluation
lm-evaluation-harness	Benchmarking L