from sentence_transformers import SentenceTransformer
import faiss
import pickle

MODEL_NAME = "all-MiniLM-L6-v2"

index = faiss.read_index("vector_store/skin_conditions.index")

with open("vector_store/documents.pkl", "rb") as f:
    documents = pickle.load(f)

model = SentenceTransformer(MODEL_NAME)


def search(query, k=3):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:
        if i < len(documents):
            results.append(documents[i])

    return results


if __name__ == "__main__":

    query = input("Enter your question: ")

    results = search(query)

    print("\nRelevant information:\n")

    for result in results:
        print("Source:", result["source"])
        print(result["text"])
        print("-" * 50)