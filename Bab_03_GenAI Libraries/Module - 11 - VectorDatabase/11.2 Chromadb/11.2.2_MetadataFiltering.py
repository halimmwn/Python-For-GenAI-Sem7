import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()

openai_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"},
)

query = "how do I add knowledge to an LLM without retraining?"
q_embed = embed([query])[0]

# Filter by metadata before similarity ranking
results_filtered = collection.query(
    query_embeddings=[q_embed],
    n_results=3,
    where={"category": {"$eq": "technique"}},  # pre-filter
    include=["documents", "distances"],
)

print("Filtered results:")
for doc, dist in zip(
    results_filtered["documents"][0], results_filtered["distances"][0]
):
    print(f"  [{dist:.4f}] {doc}")

# Update a document
collection.update(
    ids=["doc_0"],
    documents=[
        "RAG retrieves external documents and injects them into the LLM context."
    ],
    embeddings=embed([
        "RAG retrieves external documents and injects them into the LLM context."
    ]),
)

# Delete a document
collection.delete(ids=["doc_4"])
print(f"After delete, count: {collection.count()}")
