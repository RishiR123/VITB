# Day 3 – Assignment

## Prompting with GenAI APIs (Flask + Gemini)

### Objective

This assignment helps students understand how **LLM APIs work in real applications** by building and analyzing a simple Flask-based backend that communicates with the Google GenAI (Gemini) model.

---

## Task 1: Understanding the Request–Response Flow

Answer the following questions in your own words:

1. What is the role of Flask in the given application?
2. How does the user’s message reach the Gemini model?
3. What type of data is sent to the `/chat` endpoint?
4. What type of data is returned to the frontend?

---

## Task 2: Code Walkthrough

Given the provided Flask + GenAI code:

1. Explain the purpose of each of the following:

   * `genai.Client()`
   * `@app.route("/chat", methods=["POST"])`
   * `request.json["message"]`
   * `client.models.generate_content()`
   * `jsonify()`

Write your explanation clearly in short paragraphs.

---

## Task 3: Modify the Prompt

1. Modify the `/chat` route so that the prompt sent to the model is:

   * Prefixed with a role (example: "You are a helpful AI tutor")
   * Includes a clear instruction

2. Submit:

   * Updated code snippet
   * One example input
   * Corresponding model output

---

## Task 4: API Experimentation

Change **one** of the following and observe the difference:

* Model name (if available)
* Prompt wording
* Length or complexity of user input

Write a short explanation (5–6 lines) describing how the output changed.

---

## Task 5: Security & Best Practices (Short Answer)

Answer briefly:

1. Why should API keys not be hardcoded in source code?
2. Where should API keys be stored in production applications?
3. What could happen if an API key is exposed publicly?

---

## Submission Guidelines

* File name: `day3_assignment.md`
* Format: Markdown
* Include code blocks where applicable

---

## Evaluation Criteria

* Understanding of API flow
* Clarity of code explanations
* Correctness of modifications
* Awareness of security practices

---

## Deadline

Submit before the start of **Day 4 session**.
