from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ==============================================================================
# 1. Simple Template
# ==============================================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human", "{question}"),
])

# Format ke messages (untuk inspeksi)
msgs = prompt.format_messages(
    role="Python tutor",
    language="English",
    question="What is a generator?",
)

print("=== 1. Simple Formatted Messages ===")
for msg in msgs:
    print(f"[{msg.type.upper()}]: {msg.content}")

# ==============================================================================
# 2. Template dengan Conversation History Placeholder
# ==============================================================================
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today's context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

formatted = chat_prompt.format_messages(
    context="User is learning about RAG",
    history=[
        HumanMessage(content="What is retrieval?"),
    ],
    input="How is it different from fine-tuning?",
)

print("\n=== 2. Messages with History Placeholder ===")
for msg in formatted:
    print(f"[{msg.type.upper()}]: {msg.content}")

print(f"\nTotal messages in formatted prompt: {len(formatted)}")
