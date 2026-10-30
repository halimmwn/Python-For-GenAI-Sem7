import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm_oai = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
llm = llm_oai
prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise the following text in exactly {sentences} sentences."),
    ("human", "{text}"),
])
chain = prompt | llm | StrOutputParser()

input_data = {
    "sentences": 2,
    "text": (
        "LangChain is a framework for developing applications powered by large language models. "
        "It provides abstractions for prompts, models, memory, and retrieval so developers can "
        "compose complex AI workflows from simple building blocks without writing boilerplate."
    ),
}
print("=== Hasil Chain Invoke ===")
result = chain.invoke(input_data)
print(result)