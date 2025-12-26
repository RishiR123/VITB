# Day 4 – NLP Basics

## Assignment: Tokenization, Stemming, and Lemmatization using NLTK

### Objective

The goal of this assignment is to help students understand and apply core NLP preprocessing techniques:

* Sentence tokenization
* Word tokenization
* Stemming using Porter Stemmer
* Lemmatization using WordNet Lemmatizer

Students will write Python code using **NLTK** and observe how text changes at each stage.

---

## Instructions

* Use **Python 3**
* Use the **NLTK** library only
* Write clean, readable code
* Add comments explaining each step
* Save your file as `day4_nlp_basics.py`

---

## Task 1: Sentence Tokenization

1. Take the following paragraph:

```
Artificial Intelligence is transforming industries. NLP is a key part of AI. Tokenization is the first NLP step.
```

2. Use `sent_tokenize()` to split the paragraph into sentences.
3. Print the list of sentences.

Expected Outcome:

* A Python list where each element is one sentence.

---

## Task 2: Word Tokenization

1. Use the same paragraph from Task 1.
2. Apply `word_tokenize()` to split the text into words and punctuation.
3. Print the resulting list.

Expected Outcome:

* Words and punctuation should appear as separate tokens.

---

## Task 3: Stemming with Porter Stemmer

1. Create a list of words:

```
["playing", "played", "plays", "happily", "studies"]
```

2. Apply **PorterStemmer** to each word.
3. Print the original word and its stem.

Expected Outcome:

* Some stems may not be real English words.

---

## Task 4: Lemmatization with WordNet

1. Use the following list of words:

```
["running", "cars", "better", "children", "feet"]
```

2. Apply **WordNetLemmatizer** to each word.
3. Print the original word and its lemma.

Expected Outcome:

* Lemmas should be meaningful dictionary words.

---

## Task 5: Stemming vs Lemmatization Comparison

1. Choose **any five words** of your choice.
2. Apply both:

   * Porter Stemmer
   * WordNet Lemmatizer
3. Print results in the following format:

```
Word: running | Stem: run | Lemma: running
```

---

## Bonus Task (Optional)

* Try lemmatizing verbs by passing POS tags (`pos='v'`) to the lemmatizer.
* Observe how the output changes.

---

## Submission Guidelines

* Submit the file: `day4_nlp_basics.py`
* Ensure code runs without errors
* Output must match task expectations

---

## Learning Outcome

After completing this assignment, students should be able to:

* Explain the difference between tokenization, stemming, and lemmatization
* Implement basic NLP preprocessing pipelines
* Choose the correct preprocessing technique for an NLP task
