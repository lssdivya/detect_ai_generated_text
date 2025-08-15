AI-Generated Text Detector

This project detects whether a piece of text is AI-generated or human-written by combining advanced NLP embeddings, linguistic analysis, and statistical scoring. It targets AI models like GPT-3 and beyond, leveraging semantic similarity, readability metrics, and perplexity-based evaluation.

🚀 Key Features

BERT Embeddings — Generates semantic and contextual embeddings for the text.

Linguistic Analysis — Evaluates:

Flesch Reading Ease 📖

Flesch–Kincaid Grade Level 🎓

Gunning Fog Index 🌫️

Word frequency distributions 📊

Pinecone Vector Database — Efficiently stores and retrieves embeddings.

Cosine Similarity — Measures similarity between input text and known AI-written samples.

Perplexity & Burstiness Scoring — Uses OpenAI’s API for statistical text behavior analysis.

Interactive Streamlit App — User-friendly interface to test and visualize results.

🛠 Methodology

Embedding Generation — Convert input text into high-dimensional BERT embeddings capturing deep semantic meaning.

Linguistic Feature Extraction — Compute readability scores, sentence complexity, and word distribution patterns.

Vector Similarity Search — Compare embeddings with a curated set of AI-generated and human-written text using cosine similarity.

Statistical Analysis — Calculate perplexity and burstiness to assess text predictability and variability.

Classification — Feed combined features into a classifier to label the text as AI-generated or human-written.

📦 Tech Stack

Language Model: BERT

Database: Pinecone Vector DB

Similarity Metric: Cosine Similarity

Readability Metrics: Flesch, Flesch–Kincaid, Gunning Fog

API: OpenAI for perplexity & burstiness

UI: Streamlit