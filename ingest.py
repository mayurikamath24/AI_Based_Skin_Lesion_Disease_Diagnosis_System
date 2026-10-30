from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle

DOCUMENTS_PATH = Path("documents")
VECTOR_STORE_PATH = Path("vector_store")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents():
    documents = []

    for file in DOCUMENTS_PATH.glob("*.txt"):
        text = file.read_text(encoding="utf-8")

        documents.append({
            "filename": file.name,
            "text": text
        })

    return documents


def create_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def main():
    print("Loading documents...")

    documents = load_documents()

    chunks = []

    for document in documents:
        document_chunks = create_chunks(document["text"])

        for chunk in document_chunks:
            chunks.append({
                "text": chunk,
                "source": document["filename"]
            })

    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    VECTOR_STORE_PATH.mkdir(exist_ok=True)

    faiss.write_index(
        index,
        str(VECTOR_STORE_PATH / "skin_conditions.index")
    )

    with open(
        VECTOR_STORE_PATH / "documents.pkl",
        "wb"
    ) as f:
        pickle.dump(chunks, f)

    print("Vector database created successfully!")
    print(f"Stored {len(chunks)} documents.")


if __name__ == "__main__":
    main()