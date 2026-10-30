from sentence_transformers import SentenceTransformer
import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os
import re

from backend.kg import get_condition_context, normalize_code, CONDITIONS

# Resolve paths relative to this file, so it works no matter
# which folder Flask / Python is started from.
RAG_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------
# Load RAG components
# -----------------------------

index = faiss.read_index(
    os.path.join(RAG_DIR, "vector_store", "skin_conditions.index")
)

with open(
    os.path.join(RAG_DIR, "vector_store", "documents.pkl"),
    "rb"
) as f:
    documents = pickle.load(f)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# Load LLM
# -----------------------------

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

model.eval()


# -----------------------------
# Retrieve information
# -----------------------------

def retrieve(question):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    k = min(1, len(documents))

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for i in indices[0]:

        if 0 <= i < len(documents):
            results.append(documents[i])

    return results


# -----------------------------
# Clean up the model output
# -----------------------------

def clean_answer(text):
    """Strip markdown, drop a dangling cut-off fragment."""

    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"(?m)^\s*#+\s*", "", text)          # headings
    text = re.sub(r"(?m)^\s*[\*\-\u2022]\s+", "", text)   # bullets
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Remove a trailing list marker like "3." left by truncation
    text = re.sub(r"(?:\n|\s)+\d+[.)]\s*$", "", text).strip()

    # If the output was cut mid-sentence, keep only whole sentences
    if text and text[-1] not in ".!?":
        cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if cut > 0:
            text = text[:cut + 1]

    return text.strip()


# -----------------------------
# Generate answer
# -----------------------------

def generate_answer(question, condition=None, uncertain=False, extra_note=None):
    """
    question  : the user's question (or a default like
                "Explain this result.")
    condition : predicted class code or name, e.g. "mel" / "Melanoma"
                (for an Uncertain result, pass the top guess)
    uncertain : True if the app flagged the result as Uncertain
    extra_note: optional extra fact, e.g. "Only 2 of 4 AI models agreed."
    """

    code = normalize_code(condition)

    # Add the condition name to the search text so the vector
    # retrieval finds the right document for this prediction.
    search_text = question
    if code:
        search_text = CONDITIONS[code][0] + ". " + question

    results = retrieve(search_text)

    text_context = ""

    for result in results:
        text_context += result["text"] + "\n"

    graph_context = get_condition_context(
        condition,
        uncertain=uncertain
    )

    if extra_note:
        graph_context = (graph_context + "\n" + extra_note).strip()

    prompt = f"""Use only the facts below. Do not add any other medical facts.

Facts:
{graph_context or "None available."}

Reference text:
{text_context or "None available."}

Task: {question}

Rules: Write 3 to 4 short sentences of plain text. No markdown, no asterisks, no headings, no lists. Do not state a diagnosis; say this is only an AI estimate. End by advising the person to see a qualified healthcare professional."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful medical information assistant. "
                "You write short plain-text answers with no formatting symbols."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False
        )

    new_tokens = output[
        0
    ][inputs["input_ids"].shape[1]:]

    answer = clean_answer(
        tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )
    )

    # If the small model produced nothing usable, fall back to the
    # curated graph facts so the person still gets grounded text.
    if len(answer) < 40 and graph_context:
        answer = (
            graph_context
            + "\nThis is only an AI estimate. Please see a qualified "
            "healthcare professional."
        )

    return answer


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":

    condition = input(
        "Predicted condition code or name (e.g. mel), blank for none: "
    ).strip() or None

    uncertain = input(
        "Uncertain result? (y/n): "
    ).strip().lower() == "y"

    question = input(
        "Enter your question: "
    )

    answer = generate_answer(
        question,
        condition=condition,
        uncertain=uncertain
    )

    print("\nFinal Answer:\n")
    print(answer)