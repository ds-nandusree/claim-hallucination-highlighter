# Claim-Level Hallucination Highlighter

An NLP research prototype for identifying potential hallucinations in AI-generated answers through claim-level evidence retrieval and Natural Language Inference (NLI).

<p align="center">
  <a href="https://claim-hallucination-highlighter-k4pnitrf9m6xiqfpvebu2s.streamlit.app/">
    <strong>🚀 Try the Live Application</strong>
  </a>
  &nbsp; | &nbsp;
  <a href="https://github.com/ds-nandusree/claim-hallucination-highlighter">
    <strong>💻 View Source Code</strong>
  </a>
</p>

## Overview

Large Language Models (LLMs) can generate fluent responses containing inaccurate, unsupported, or fabricated information. Evaluating an entire response as a single unit can make it difficult to identify which individual claims require verification.

The **Claim-Level Hallucination Highlighter** addresses this challenge by breaking AI-generated answers into individual claims, retrieving relevant evidence from a curated collection of 30 landmark articles, and evaluating the relationship between each claim and its retrieved evidence using a Natural Language Inference (NLI) model.

The application presents claim-level classifications, relevant evidence, and model scores to help users inspect potentially unreliable statements.

## Key Features

- **Claim-Level Analysis:** Examines individual statements rather than evaluating an entire answer as one unit.
- **Semantic Evidence Retrieval:** Retrieves relevant text passages using dense vector embeddings and FAISS similarity search.
- **NLI-Based Verification:** Uses a pretrained DeBERTa model to estimate whether retrieved evidence supports or contradicts a claim.
- **Three-Way Classification:** Categorizes claims as Supported, Contradicted, or Not Enough Information.
- **Evidence Transparency:** Displays retrieved passages and model scores to help users inspect the basis of each prediction.
- **Interactive Web Interface:** Provides an accessible interface built with Streamlit.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web Interface | Streamlit |
| Sentence Embeddings | Sentence Transformers — `all-MiniLM-L6-v2` |
| Evidence Retrieval | FAISS |
| Natural Language Inference | DeBERTa — `cross-encoder/nli-deberta-v3-base` |
| Text Processing | NLTK |
| Evidence Collection | Curated collection of 30 landmark articles |

## System Workflow

1. **Input:** The user provides an AI-generated answer for analysis.
2. **Claim Extraction:** The answer is split into individual sentences or claims.
3. **Embedding Generation:** Sentence Transformers converts claims into semantic vector representations.
4. **Evidence Retrieval:** FAISS searches the indexed text chunks and retrieves relevant passages.
5. **NLI Evaluation:** The DeBERTa model evaluates the relationship between each claim and its retrieved evidence.
6. **Classification:** The system assigns a Supported, Contradicted, or Not Enough Information classification.
7. **Results:** The interface displays the classifications, retrieved evidence, and model scores.

## Project Architecture

The system combines semantic search with evidence-based claim evaluation. The retrieval component identifies potentially relevant passages, while the NLI component estimates the relationship between the retrieved evidence and each claim.

```text
AI-Generated Answer
        |
        v
Claim Segmentation
        |
        v
Sentence Transformers
(all-MiniLM-L6-v2)
        |
        v
FAISS Evidence Retrieval
        |
        v
Retrieved Evidence + Claim
        |
        v
DeBERTa NLI Model
        |
        v
Claim-Level Classification
        |
        v
Evidence and Scores in Streamlit
```

## Repository Structure

```text
claim-hallucination-highlighter/
├── app.py
├── requirements.txt
├── chunks.json
├── chunks.faiss
├── answers.json
└── config.json
```

- `app.py` — Streamlit application and claim-analysis workflow.
- `requirements.txt` — Python dependency definitions.
- `chunks.json` — Text chunks used for evidence retrieval.
- `chunks.faiss` — FAISS index for semantic similarity search.
- `answers.json` — Supporting answer-related data used by the application.
- `config.json` — Application configuration.

## Getting Started

### Prerequisites

- Python installed on your system
- Git
- A stable internet connection for downloading pretrained models when they are not already cached

### 1. Clone the Repository

```bash
git clone https://github.com/ds-nandusree/claim-hallucination-highlighter.git
```

### 2. Navigate to the Project Directory

```bash
cd claim-hallucination-highlighter
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## Evaluation and Research Scope

The prototype was evaluated using a small, manually reviewed set of claims. In the preliminary evaluation, the retrieval-plus-NLI approach achieved a binary F1 score of **0.75** on a 50-claim test set. Its three-class test accuracy was **0.56**.

These results are preliminary and should not be interpreted as evidence of production-level reliability. Performance may vary depending on claim complexity, evidence retrieval quality, model predictions, and the available corpus.

## Limitations

- **Limited knowledge base:** Verification is restricted to the available collection of 30 landmark articles.
- **Retrieval dependency:** Relevant evidence may not be retrieved even when it exists in the source collection.
- **Context sensitivity:** Complex claims containing multiple facts may receive misleading classifications when the retrieved passage addresses only part of the claim.
- **Model uncertainty:** NLI scores are model outputs, not guaranteed probabilities of factual correctness.
- **Unverified does not mean false:** A claim classified as Not Enough Information may be true, but the retrieved evidence does not establish it sufficiently.

This application is an experimental research prototype and should support, rather than replace, independent fact-checking.

## Future Enhancements

- Expand the evidence collection with additional trusted sources.
- Improve claim segmentation for compound and complex statements.
- Add source citations and evidence-ranking improvements.
- Evaluate performance on larger, independently annotated datasets.
- Improve confidence calibration and handling of ambiguous evidence.

## Learning Outcomes

This project explores practical applications of Natural Language Processing, semantic search, vector indexing, pretrained language models, Natural Language Inference, and interactive machine learning applications.

It also demonstrates how information retrieval and language-model reasoning can be combined to make AI-generated answers easier to inspect and verify.

## Author

**Diguvasadum Nandusree**

B.Tech — Computer Science and Engineering

- **GitHub:** https://github.com/ds-nandusree
- **Portfolio:** https://ds-nandusree.github.io/portfolio/
- **Live Project:** https://claim-hallucination-highlighter-k4pnitrf9m6xiqfpvebu2s.streamlit.app/

---

*Built as an NLP research prototype exploring evidence-based verification of AI-generated claims.*
