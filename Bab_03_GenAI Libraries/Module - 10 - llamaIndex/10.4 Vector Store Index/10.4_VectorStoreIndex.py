import os
from dotenv import load_dotenv

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_tokens=1024,
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    api_base="https://openrouter.ai/api/v1",
)

documents = [
    Document(
        text="RAG combines retrieval with generation to answer questions from external knowledge."
    ),
    Document(
        text="Fine-tuning trains a model on new data to improve performance on a specific task."
    ),
    Document(
        text="Prompt engineering optimises the input to an LLM to get better outputs without any training."
    ),
    Document(
        text="Vector databases store high-dimensional embeddings and support fast approximate nearest-neighbour search."
    ),
    Document(
        text="LangChain provides abstractions for building chains, agents, and memory-enabled LLM applications."
    ),
]

# Build index - embeds all documents automatically
index = VectorStoreIndex.from_documents(documents)

# Query engine - retrieves + synthesises answer
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What is the difference between RAG and fine-tuning?")

print("Answer:", response)
print("\nSource nodes:")
for node in response.source_nodes:
    print(f"  Score {node.score:.4f}: {node.text[:60]}...")
