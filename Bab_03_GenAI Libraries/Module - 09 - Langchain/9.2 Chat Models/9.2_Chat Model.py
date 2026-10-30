import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ==============================================================================
# 1. Setup Model OpenAI via OpenRouter
# ==============================================================================
llm_oai = ChatOpenAI(
    model="openai/gpt-4o",  # atau model lain seperti "openai/gpt-4o-mini"
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

# ==============================================================================
# 2. Setup Model Anthropic via XKiro
# ==============================================================================
llm_ant = ChatAnthropic(
    model="qwen/qwen3.7-flash:free",  # atau "anthropic/claude-sonnet-4.5"
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ.get("XKIRO_API_KEY"),
    base_url="https://api.xkiro.com",
)

# ==============================================================================
# 3. Interaksi dengan Model (Pesan Standar LangChain)
# ==============================================================================
messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain Expression Language?"),
]

# Pilih model yang ingin dijalankan (llm_ant atau llm_oai)
selected_llm = llm_ant

# 1. Invoke (Blocking)
print("=== 1. Invoke (Blocking) ===")
try:
    response = selected_llm.invoke(messages)
    print(response.content)
except Exception as e:
    print(f"Invoke error: {e}")

# 2. Stream
print("\n=== 2. Stream ===")
try:
    for chunk in selected_llm.stream(messages):
        print(chunk.content, end="", flush=True)
    print()
except Exception as e:
    print(f"\nStream error: {e}")

# 3. Batch (Parallel)
print("\n=== 3. Batch (Parallel) ===")
try:
    responses = selected_llm.batch([
        [HumanMessage(content="What is RAG?")],
        [HumanMessage(content="What is a vector database?")],
    ])
    for idx, r in enumerate(responses, 1):
        print(f"\n[Pertanyaan {idx}]:\n{r.content[:120]}...\n")
except Exception as e:
    print(f"Batch error: {e}")
