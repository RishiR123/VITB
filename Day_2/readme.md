# Day 2 – Prompting Basics

## What is Prompting

Prompting is the process of giving **instructions and input** to a Large Language Model (LLM) to guide its output. The quality, clarity, and structure of a prompt directly affect the quality of the response.

A prompt can include:

* Instruction (what to do)
* Context (background information)
* Input data (text, question, code)

---

## Zero-Shot Prompting

Zero-shot prompting means asking the model to perform a task **without providing any examples**.

### Characteristics

* Relies entirely on the model’s pretrained knowledge
* Simple and fast to use
* Output quality depends heavily on prompt clarity

### Example

Prompt:

```
Explain what Natural Language Processing is.
```

Use-cases:

* Simple explanations
* General questions
* Broad reasoning tasks

---

## Few-Shot Prompting

Few-shot prompting provides **examples inside the prompt** to guide the model’s behavior.

### Characteristics

* Improves consistency and accuracy
* Helps enforce format and style
* Slightly increases token usage

### Example

Prompt:

```
Translate English to French:
English: Hello
French: Bonjour

English: Thank you
French:
```

Use-cases:

* Structured outputs
* Classification tasks
* Format-sensitive responses

---

## Chain-of-Thought Prompting

Chain-of-Thought (CoT) prompting encourages the model to **reason step-by-step** before producing the final answer.

### Why Chain-of-Thought Matters

* Improves reasoning accuracy
* Helps with math, logic, and multi-step problems
* Makes outputs more explainable

### Example

Prompt:

```
Solve the problem step by step:
If a shirt costs 500 and there is a 20% discount, what is the final price?
```

---

## Prompt Components (Best Practice)

A well-structured prompt usually contains:

1. Role
2. Task
3. Constraints
4. Output format

Example:

```
You are a data science instructor.
Explain overfitting in simple terms.
Limit the answer to 5 bullet points.
```

---

## Prompting using Google GenAI Direct API

### What is Google GenAI API

Google GenAI API allows developers to interact with Gemini models programmatically using HTTP or SDKs.

---

### Basic Python Example (Direct Prompting)

```python
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain zero-shot and few-shot prompting"
)

print(response.text)
```

---

## Zero-Shot Prompt via API

```python
prompt = "Summarize the concept of transformers in simple words"

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print(response.text)
```

---

## Few-Shot Prompt via API

```python
prompt = """
Classify sentiment:
Text: I love this product
Sentiment: Positive

Text: This is the worst service
Sentiment: Negative

Text: The movie was okay
Sentiment:
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print(response.text)
```

---

## Chain-of-Thought Prompt via API

```python
prompt = "Solve step by step: A train travels 60 km in 1 hour. How far will it travel in 3 hours?"

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print(response.text)
```

---

## Common Prompting Mistakes

* Vague instructions
* Missing context
* Overloading prompts with unnecessary text
* Ignoring output format

---

## Learning Outcome of Day 2

By the end of Day 2, students should be able to:

* Understand zero-shot, few-shot, and chain-of-thought prompting
* Design clear and structured prompts
* Use Google GenAI API for direct prompting
* Compare outputs across prompting techniques
