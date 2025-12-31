## Day 7 – ATS (Applicant Tracking System) Project using Gemini GenAI (PDF Parsing)

---

### Overview

An Applicant Tracking System (ATS) is used by companies to automatically analyze resumes and match them with job descriptions. In this project, students build a **mini ATS system** using **Google Gemini GenAI** that can:

* Accept a **resume PDF**
* Extract text from the PDF
* Parse resume content using an LLM
* Parse job description
* Compare resume and job description
* Generate an AI-based match score and feedback

This project demonstrates **real-world usage of Generative AI** in HR and recruitment systems.

---

### Problem Statement

Manual resume screening is:

* Time-consuming
* Error-prone
* Not scalable

The goal is to automate this process using **GenAI-powered reasoning**.

---

### System Flow

1. User uploads `resume.pdf`
2. Backend extracts text from PDF
3. Resume content is parsed using Gemini
4. Job description is parsed using Gemini
5. ATS logic compares both inputs
6. AI-generated evaluation is returned as JSON

---

### Technology Stack

* Python
* Flask
* PyPDF2 (PDF text extraction)
* Google Gemini GenAI API
* JSON

---

### Resume Parsing

Resume parsing means converting **unstructured resume text** into structured information such as:

* Skills
* Experience
* Education
* Tools & technologies

Instead of traditional regex-based parsing, this project uses **LLM-based parsing**, which is:

* More flexible
* Context-aware
* Industry-aligned

---

### PDF Text Extraction

PDF resumes cannot be directly processed by LLMs. Therefore:

* Resume is uploaded as a PDF file
* PyPDF2 is used to extract raw text
* Extracted text is passed to Gemini for parsing

---

### Job Description Parsing

The job description is parsed to extract:

* Required skills
* Responsibilities
* Preferred qualifications

This ensures fair and structured comparison with the resume.

---

### ATS Matching Logic

The ATS system compares:

* Parsed resume details
* Parsed job description requirements

Gemini generates:

* Match percentage (0–100)
* Matching skills
* Missing skills
* Strengths
* Improvement suggestions

---

### API Design

The Flask backend exposes a single API endpoint:

`POST /analyze`

The endpoint:

* Accepts resume PDF via `multipart/form-data`
* Accepts job description as text
* Returns structured ATS evaluation

---

### Sample Input

* Resume: `resume.pdf`
* Job Description: Backend developer with Python, NLP, and API experience

---

### Sample Output

* Match Percentage: 85–90%
* Matching Skills: Python, NLP, APIs
* Missing Skills: Cloud deployment
* Suggestions: Add cloud-based project experience

---

### Key Concepts Covered

* File upload handling in Flask
* PDF parsing
* LLM-based resume parsing
* Prompt engineering
* Real-world GenAI workflows
* ATS system design

---

### Learning Outcomes

By the end of Day 7, students will be able to:

* Build a complete GenAI-based project
* Parse unstructured documents
* Use Gemini API for reasoning tasks
* Design production-style Flask APIs
* Understand ATS systems used in industry

---

### Optional Enhancements

* Convert outputs to strict JSON format
* Add embeddings for semantic matching
* Build frontend UI for resume upload
* Store ATS results in database
* Extend system into RAG-based ATS

---

### Summary

This ATS project combines **PDF parsing**, **LLM reasoning**, and **backend API development** to simulate a real-world GenAI application. It prepares students for building **production-grade AI systems** used across industries.
