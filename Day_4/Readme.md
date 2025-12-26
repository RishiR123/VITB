# Day 4 – NLP Basics (Student Notes)

## 1. Sentence Tokenization

Sentence tokenization splits a paragraph into individual sentences.

```python
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')

text = "NLP is powerful. It is used in chatbots. Tokenization is the first step."
sentences = sent_tokenize(text)
print(sentences)
```

---

## 2. Word Tokenization

Word tokenization splits text into individual words and punctuation.

```python
from nltk.tokenize import word_tokenize

text = "NLP is powerful. It is used in chatbots."
words = word_tokenize(text)
print(words)
```

---

## 3. Porter Stemmer

Porter Stemmer reduces words to their root form by removing suffixes. It is fast but not linguistically accurate.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "runs", "runner", "easily"]

stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words)
```

---

## 4. WordNet Lemmatizer

WordNet Lemmatizer converts words to their meaningful base form using a dictionary.

```python
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
words = ["running", "cars", "better", "feet"]

lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print(lemmatized_words)
```

---

## Summary

* Sentence tokenization splits text into sentences
* Word tokenization splits text into words
* Porter Stemmer is fast but may produce incorrect roots
* WordNet Lemmatizer gives meaningful base words and is preferred for NLP tasks
