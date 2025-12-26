# Day 3 – Prompting with GenAI APIs

## Overview

Day 3 focuses on **using Large Language Models through APIs** and understanding how prompts flow from a frontend or client to an LLM and back as a response. Students will learn how prompts are sent, how responses are generated, and how to build a simple **Flask-based chat backend** using the Google GenAI (Gemini) API.

---

## What are LLM APIs

LLM APIs allow applications to interact with large language models programmatically.

Instead of manually typing prompts into a chat interface, applications send prompts as **HTTP requests** and receive generated responses as **JSON data**.

Key benefits:

* Automation
* Integration into web apps
* Scalability
* Controlled usage

---

## Request–Response Lifecycle

1. User sends input from UI or client
2. Backend receives the request
3. Prompt is sent to the LLM API
4. LLM processes the prompt
5. Generated response is returned
6. Backend sends response back to the client

This lifecycle is the foundation of chatbots, assistants, RAG systems, and agents.

---

## Google GenAI (Gemini) API

Google GenAI provides access to **Gemini models** through a Python SDK.

Core components:

* Client initialization with API key
* Model selection
* Prompt (contents)
* Generated response

---

## Flask as a Backend Server

Flask is a lightweight Python web framework used to build APIs quickly.

In this setup:

* Flask acts as the backend server
* Gemini acts as the intelligence layer
* JSON is used for communication

---

## Example: Flask + Gemini Chat API

### Code Explanation

```python
from flask import Flask, request, jsonify, render_template
from google import genai

# Initialize GenAI client (API key should be stored securely)
client = genai.Client(api_key="YOUR_API_KEY")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Get user message from request
    prompt = request.json["message"]

    # Send prompt to Gemini model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # Return model response
    return jsonify({"reply": response.text})

app.run(port=8080)
```

---

## Understanding the Code Flow

* `Flask()` initializes the backend server
* `/` route serves the frontend (HTML page)
* `/chat` route receives user input via POST request
* Prompt is sent to Gemini using `generate_content()`
* Model output is returned as JSON

---

## Prompting through APIs vs Chat UI

| Aspect         | Chat UI | API Prompting |
| -------------- | ------- | ------------- |
| Manual         | Yes     | No            |
| Automation     | Limited | High          |
| Integration    | Low     | High          |
| Production Use | No      | Yes           |

---

## Important API Parameters (Conceptual)

* Model: Defines capability and speed
* Prompt (contents): Input text
* Tokens: Output length control
* Temperature: Creativity vs determinism

These parameters become critical in production systems.

---

## Security Best Practices

* Never hardcode API keys in source code
* Use environment variables
* Do not expose keys in frontend code

Example:

```bash
export GENAI_API_KEY="your_api_key"
```

---

## Real-World Use Cases

* Web-based chatbots
* AI-powered customer support
* Prompt-based internal tools
* Backend for RAG and agent systems

---

## Learning Outcome of Day 3

By the end of Day 3, students should be able to:

* Understand LLM API request–response flow
* Use Google GenAI API programmatically
* Build a basic Flask-based chat backend
* Connect prompts to real applications
