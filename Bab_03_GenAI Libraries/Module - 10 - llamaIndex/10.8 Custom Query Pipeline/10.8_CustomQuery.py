import os
from dotenv import load_dotenv

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
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
        text="HNSW (Hierarchical Navigable Small World) is the graph algorithm powering most modern vector search."
    ),
    Document(
        text="IVF (Inverted File Index) partitions vectors into clusters for fast approximate search."
    ),
    Document(
        text="PQ (Product Quantisation) compresses vectors to reduce memory usage at the cost of recall."
    ),
    Document(
        text="Exact nearest-neighbour search guarantees correctness but is O(n) - unusable at scale."
    ),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

# Different response modes
for mode in [ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE]:
    synthesiser = get_response_synthesizer(response_mode=mode)
    engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesiser,
    )
    resp = engine.query("What algorithms are used in vector search?")
    print(f"[{mode}] {str(resp)[:120]}\n")
