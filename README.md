# 40-Day GenAI & Agentic AI Program – Detailed Notes

Each day represents ~3 hours of learning: **Concepts (1 hr) + Hands-on (1.5 hrs) + Reflection/Assignment (0.5 hr)**. These notes are meant to go inside each day's `notes.md` file in the Git repository.

---

## Day 1 – Introduction to Generative AI

Generative AI refers to models that can create new content such as text, images, audio, or code. Traditional AI focused on prediction and classification, while GenAI focuses on generation. The evolution includes Rule-based systems → ML → Deep Learning → Transformers → LLMs.

Key use cases include chatbots, content generation, code assistants, search, recommendations, and autonomous agents. The GenAI ecosystem includes foundation models, APIs, tooling layers, vector databases, agent frameworks, and applications.

---

## Day 2 – Prompting Basics

Prompting is the process of instructing an LLM to perform a task. A prompt usually contains instruction, context, and input.

Zero-shot prompting means no examples are provided. Few-shot prompting includes examples to guide behavior. Prompt clarity directly impacts output quality. Prompt engineering is iterative and experimental.

---

## Day 3 – Prompting with APIs

LLMs are accessed using APIs via HTTP requests. A typical request contains the model name, prompt, temperature, and token limits. Responses include generated text and metadata.

Understanding request/response flow is critical for building applications. Token usage affects cost and latency. Error handling and retries are essential for production systems.

---

## Day 4 – NLP Basics

Natural Language Processing prepares text for machine understanding. Tokenization splits text into sentences or words. Stemming reduces words to crude roots. Lemmatization converts words into meaningful base forms using vocabulary and grammar.

These steps form the foundation of all downstream NLP, embeddings, and LLM pipelines.

---

## Day 5 – Embeddings

Embeddings are numerical vector representations of text that capture semantic meaning. Similar texts have vectors close in vector space.

Cosine similarity is commonly used to measure closeness. Embeddings enable semantic search, clustering, recommendation, and RAG systems.

---

## Day 6 – Transformers

Transformers are neural network architectures based on attention mechanisms. Attention allows the model to focus on relevant parts of input.

Key components include embeddings, positional encoding, multi-head attention, feed-forward layers, and residual connections. Transformers outperform RNNs in scalability and parallelism.

---

## Day 7 – LLM Evaluation

LLMs can hallucinate, produce biased outputs, or generate incorrect information. Evaluation focuses on factuality, coherence, relevance, and safety.

Mitigation strategies include grounding with data, prompt constraints, and retrieval-based systems.

---

## Day 8 – Mini NLP Lab

This day focuses on combining tokenization, stopword removal, stemming, and lemmatization into a preprocessing pipeline.

Students clean raw text and prepare it for embeddings or modeling tasks.

---

## Day 9 – Prompt Patterns

Prompt patterns include role prompting, step-by-step reasoning, and Chain-of-Thought prompting. These patterns improve reasoning and structure.

Understanding when to use explicit reasoning prompts is critical for complex tasks.

---

## Day 10 – Mini Project Review 1

Students build a small prompt-based application such as a summarizer or Q&A bot. Focus is on prompt quality, not architecture.

Peer review emphasizes clarity, correctness, and creativity.

---

## Day 11 – RAG Introduction

Retrieval-Augmented Generation combines external knowledge with LLMs. Instead of relying only on model memory, relevant documents are retrieved and passed as context.

RAG reduces hallucinations and improves accuracy.

---

## Day 12 – Vector Databases

Vector databases store embeddings and allow fast similarity search. FAISS and Chroma are commonly used.

They index vectors efficiently and enable scalable semantic retrieval.

---

## Day 13 – Chunking

Documents must be split into chunks before embedding. Chunk size affects retrieval quality.

Overlap, recursive splitting, and semantic splitting are common strategies.

---

## Day 14 – RAG Pipeline

A RAG pipeline includes document loading, chunking, embedding, storage, retrieval, and generation.

The retriever selects relevant chunks which are injected into the LLM prompt.

---

## Day 15 – RAG Evaluation

RAG systems are evaluated on relevance, grounding, and completeness.

Metrics include answer correctness, context relevance, and faithfulness.

---

## Day 16 – Advanced RAG

Advanced RAG includes query routing, hybrid search, and multi-retriever setups.

Different queries may require different data sources or retrieval strategies.

---

## Day 17 – Context Optimization

Context windows are limited. Techniques such as summarization, compression, and ranking help fit relevant information.

Efficient context usage reduces cost and improves performance.

---

## Day 18 – RAG with UI

RAG systems are exposed through UIs using tools like Streamlit.

Focus is on user experience, latency, and streaming responses.

---

## Day 19 – Mini Project 2

Students build a full RAG application using their own dataset.

Evaluation focuses on correctness, architecture, and usability.

---

## Day 20 – Deployment

Applications are deployed using FastAPI. Concepts include routes, request models, and responses.

Deployment introduces API design and production considerations.

---

## Day 21 – Agentic AI Introduction

Agents are systems that can plan, act, observe, and iterate.

They extend LLMs with autonomy and goal-driven behavior.

---

## Day 22 – Agent Architecture

Agent components include planners, tools, memory, and executors.

Well-designed agents separate reasoning from execution.

---

## Day 23 – LangChain Agents

LangChain provides abstractions for building tool-using agents.

Students implement agents that call APIs and tools.

---

## Day 24 – LangGraph

LangGraph models agents as state machines.

This enables deterministic control, loops, and branching.

---

## Day 25 – CrewAI

CrewAI enables multi-agent collaboration.

Agents are assigned roles and tasks to solve problems collectively.

---

## Day 26 – AutoGen

AutoGen focuses on agent-to-agent conversations.

Useful for collaborative reasoning and negotiation workflows.

---

## Day 27 – Memory Systems

Agents require memory for continuity. Short-term memory handles context, long-term memory stores knowledge.

Vector memory and databases are commonly used.

---

## Day 28 – Tool Use

Agents interact with external systems via tools such as APIs and databases.

Tool reliability and security are critical.

---

## Day 29 – Research Agent

Research agents gather, summarize, and synthesize information.

They combine search, retrieval, and reasoning.

---

## Day 30 – Automation Agent

Automation agents perform repetitive tasks such as scheduling, reporting, and workflows.

Focus is on reliability and idempotency.

---

## Day 31 – Data Agent

Data agents analyze datasets, generate insights, and visualize results.

They integrate LLMs with data processing libraries.

---

## Day 32 – Multi-Agent Systems

Multi-agent systems coordinate multiple agents with shared goals.

Challenges include communication, conflict resolution, and orchestration.

---

## Day 33 – Safety & Guardrails

Safety includes prompt filtering, output validation, and access control.

Guardrails prevent misuse and unsafe outputs.

---

## Day 34 – Cost Optimization

Token usage directly impacts cost.

Optimization includes prompt compression, caching, and batching.

---

## Day 35 – Monitoring & Evaluation

Production systems require logging, tracing, and evaluation.

Monitoring helps detect failures and degradation.

---

## Day 36 – Final Project Planning

Students define problem statements, architecture, and milestones.

Focus is on feasibility and impact.

---

## Day 37 – Hackathon Day 1

Teams build core functionality.

Mentors guide architecture and debugging.

---

## Day 38 – Hackathon Day 2

Teams finalize features and polish UI.

Testing and optimization are prioritized.

---

## Day 39 – Demo Preparation

Students prepare demos and pitches.

Emphasis on clarity, value proposition, and live reliability.

---

## Day 40 – Final Demo & Evaluation

Teams present projects.

Evaluation considers technical depth, innovation, and execution.
