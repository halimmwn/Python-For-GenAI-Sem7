from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
    "HTTP-Referer": "https://github.com/",
    "X-Title": "Python For GenAI",
    },
)
message = client.chat.completions.create(
model="openai/gpt-4o-mini",
messages=[
    {"role": "system", "content": "You are a concise technical writer. Answer in plain English, no jargon."},
    {"role": "user", "content": "Explain what a vector database does."}
    ]
)
print(message.choices[0].message.content)