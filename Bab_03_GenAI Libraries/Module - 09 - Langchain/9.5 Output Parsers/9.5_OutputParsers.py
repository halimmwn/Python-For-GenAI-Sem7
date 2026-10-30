import os
from typing import List
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# Setup model via OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # atau "openai/gpt-4o"
    max_tokens=1024,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


# Pydantic schema for structured output
class ModelComparison(BaseModel):
    models: List[str] = Field(description="Model names compared")
    best_for_cost: str = Field(description="Most cost-effective model")
    best_for_quality: str = Field(description="Highest quality model")
    recommendation: str = Field(description="Overall recommendation")


# Use with_structured_output for schema-enforced responses (OpenAI)
structured_llm = llm.with_structured_output(ModelComparison)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI model comparison expert."),
    ("human", "Compare these models for a startup on a budget: {models}"),
])

chain = prompt | structured_llm

result: ModelComparison = chain.invoke({
    "models": "gpt-4o, gpt-4o-mini, claude-sonnet-4-5"
})

print(f"Best for cost: {result.best_for_cost}")
print(f"Best for quality: {result.best_for_quality}")
print(f"Recommendation: {result.recommendation}")
