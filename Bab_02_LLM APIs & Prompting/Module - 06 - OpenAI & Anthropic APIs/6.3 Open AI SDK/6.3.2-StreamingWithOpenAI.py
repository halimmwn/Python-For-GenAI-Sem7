from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
stream = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    max_tokens=512,
    stream=True,
    messages=[{"role": "user", "content": "Explain embeddings in 3 bullet points."}]
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
