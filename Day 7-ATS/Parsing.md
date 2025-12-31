# 📄 PDF Text Extraction in Python (Tutorial)

This tutorial explains how to extract text from a PDF file using Python.
It is useful for resume parsing, document analysis, ATS systems, chatbots, and RAG pipelines.

---

## 1️⃣ What We Are Doing

We will:

* Read a PDF file
* Extract text from each page
* Combine all text into a single string
* Return it for further processing (LLMs, NLP, storage)

---

## 2️⃣ Required Library

We use **PyPDF2**, a lightweight PDF reader.

### Install:

```bash
pip install PyPDF2
```

---

## 3️⃣ PDF Text Extraction Code

```python
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from all pages of a PDF file.
    """
    extracted_text = ""

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text

    return extracted_text
```

---

## 4️⃣ How It Works

* Open PDF in binary mode
* Load PDF using `PdfReader`
* Loop through all pages
* Extract text from each page
* Safely append extracted content

---

## 5️⃣ Example Usage

```python
pdf_path = "resume.pdf"
text = extract_text_from_pdf(pdf_path)

print(text[:1000])
```

---

## 6️⃣ Common Issues

### Empty Output

* PDF is scanned (image-based)
* No selectable text

Solution:

* Use OCR tools like Tesseract or AWS Textract

---

### Formatting Issues

* PDFs do not preserve logical structure
* LLMs handle this well for ATS systems

---

## 7️⃣ Why This Works Well for ATS / LLM Pipelines

After extraction, the text can be:

* Sent to Gemini / GPT / LLaMA
* Chunked and embedded
* Stored in vector databases
* Used for resume or JD parsing

---

## 8️⃣ Next Improvements

* OCR fallback
* Text cleaning
* Intelligent chunking
* Resume-specific formatting

---
