
import html
import json
from pathlib import Path

import faiss
import nltk
import numpy as np
import streamlit as st
import torch

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

st.set_page_config(
    page_title="Claim-Level Hallucination Highlighter",
    page_icon="🔎",
    layout="wide",
)

BASE = Path(__file__).parent
NLI_NAME = "cross-encoder/nli-deberta-v3-base"


@st.cache_resource(show_spinner="Loading AI models...")
def load_resources():
    nltk.download("punkt_tab", quiet=True)

    with open(BASE / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(str(BASE / "chunks.faiss"))

    with open(BASE / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    with open(BASE / "answers.json", encoding="utf-8") as f:
        answers = json.load(f)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    tokenizer = AutoTokenizer.from_pretrained(NLI_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_NAME
    )

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    id2label = {
        int(i): str(label).lower()
        for i, label in model.config.id2label.items()
    }

    return (
        chunks,
        index,
        config,
        answers,
        embedding_model,
        tokenizer,
        model,
        device,
        id2label,
    )


def get_nli_scores(premise, hypothesis, tokenizer, model, device, id2label):
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(
            model(**inputs).logits, dim=1
        )[0].cpu().tolist()

    scores = {}

    for i, probability in enumerate(probabilities):
        label = id2label[i]

        if "entail" in label:
            scores["entailment"] = probability
        elif "contrad" in label:
            scores["contradiction"] = probability
        elif "neutral" in label:
            scores["neutral"] = probability

    if "entailment" not in scores or "contradiction" not in scores:
        raise ValueError(f"Unexpected NLI model labels: {id2label}")

    return scores


def verify_claim(claim, resources):
    (
        chunks,
        index,
        config,
        answers,
        embedding_model,
        tokenizer,
        model,
        device,
        id2label,
    ) = resources

    query = embedding_model.encode(
        [claim], normalize_embeddings=True
    ).astype("float32")

    similarities, ids = index.search(query, 5)

    best_entailment = 0.0
    best_contradiction = 0.0
    entailment_chunk = None
    contradiction_chunk = None
    used = 0

    for similarity, chunk_id in zip(similarities[0], ids[0]):
        if chunk_id < 0 or similarity < config["min_sim"]:
            continue

        evidence = chunks[int(chunk_id)]["text"]

        scores = get_nli_scores(
            evidence,
            claim,
            tokenizer,
            model,
            device,
            id2label,
        )

        used += 1

        if scores["entailment"] > best_entailment:
            best_entailment = scores["entailment"]
            entailment_chunk = int(chunk_id)

        if scores["contradiction"] > best_contradiction:
            best_contradiction = scores["contradiction"]
            contradiction_chunk = int(chunk_id)

    if used == 0:
        verdict = "NOT_ENOUGH_INFO"
        evidence_id = int(ids[0][0]) if ids[0][0] >= 0 else None
    elif (
        best_entailment >= config["t_ent"]
        and best_entailment >= best_contradiction
    ):
        verdict = "SUPPORTED"
        evidence_id = entailment_chunk
    elif best_contradiction >= config["t_con"]:
        verdict = "UNSUPPORTED"
        evidence_id = contradiction_chunk
    else:
        verdict = "NOT_ENOUGH_INFO"
        evidence_id = int(ids[0][0]) if ids[0][0] >= 0 else None

    evidence_item = chunks[evidence_id] if evidence_id is not None else None

    return {
        "verdict": verdict,
        "entailment": round(best_entailment, 3),
        "contradiction": round(best_contradiction, 3),
        "similarity": round(float(similarities[0][0]), 3),
        "evidence": evidence_item,
    }


st.title("🔎 Claim-Level Hallucination Highlighter")
st.write(
    "Check an AI-generated answer against a curated collection of "
    "30 landmark articles using evidence retrieval and natural "
    "language inference (NLI)."
)

st.info(
    "This research prototype checks claims against its available corpus. "
    "A claim marked Unverified may be true but not established by this corpus."
)

try:
    resources = load_resources()
except Exception as exc:
    st.error(f"Could not load project resources: {exc}")
    st.stop()

answers = resources[3]
examples = {}

for answer in answers:
    question = answer.get("question", "Example answer")
    category = answer.get("type", "General")
    text = answer.get("greedy", "")

    if text:
        examples[f"{category}: {question}"] = text

choice = st.selectbox(
    "Choose a saved example or enter your own answer",
    ["Write my own"] + list(examples.keys()),
)

default_text = "" if choice == "Write my own" else examples[choice]

answer_text = st.text_area(
    "AI-generated answer",
    value=default_text,
    height=220,
    placeholder="Paste an AI-generated answer here...",
    key=f"answer_{choice}",
)

if st.button("Verify claims", type="primary"):
    if not answer_text.strip():
        st.warning("Please enter an answer first.")
    else:
        try:
            sentences = [
                sentence.strip()
                for sentence in sent_tokenize(answer_text.strip())
                if len(sentence.split()) >= 4
            ]

            if not sentences:
                st.warning("No sentences long enough to evaluate were found.")
            else:
                st.subheader("Claim-by-claim results")

                for number, sentence in enumerate(sentences, start=1):
                    with st.spinner(
                        f"Checking claim {number} of {len(sentences)}..."
                    ):
                        result = verify_claim(sentence, resources)

                    verdict = result["verdict"]

                    if verdict == "SUPPORTED":
                        heading = "✅ Supported"
                    elif verdict == "UNSUPPORTED":
                        heading = "❌ Likely wrong"
                    else:
                        heading = "⚠️ Unverified"

                    with st.container(border=True):
                        st.markdown(f"**Claim {number} — {heading}**")
                        st.write(sentence)

                        with st.expander("View evidence and model scores"):
                            st.write(
                                f"Entailment score: {result['entailment']}"
                            )
                            st.write(
                                "Contradiction score: "
                                f"{result['contradiction']}"
                            )
                            st.write(
                                "Best retrieval similarity: "
                                f"{result['similarity']}"
                            )

                            evidence = result["evidence"]

                            if evidence:
                                st.markdown("**Retrieved evidence**")
                                st.write(evidence.get("text", ""))
                                st.caption(
                                    f"Article: {evidence.get('article', 'Unknown')} "
                                    f"| Section: {evidence.get('section', 'Unknown')}"
                                )
                            else:
                                st.write("No evidence chunk was retrieved.")

        except Exception as exc:
            st.error(f"Claim verification failed: {exc}")
