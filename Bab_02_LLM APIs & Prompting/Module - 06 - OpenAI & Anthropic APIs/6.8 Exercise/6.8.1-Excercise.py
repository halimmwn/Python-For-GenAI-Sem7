import os
import time
import asyncio
import pandas as pd
import anthropic
import openai
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# API Configurations
# =========================================================
# Anthropic via xKiro
anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("XKIRO_API_KEY"),
    base_url="https://api.xkiro.com"
)
async_anthropic_client = anthropic.AsyncAnthropic(
    api_key=os.environ.get("XKIRO_API_KEY"),
    base_url="https://api.xkiro.com"
)

# OpenAI via OpenRouter
openai_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
async_openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# =========================================================
# 1. Retry on Rate Limit
# =========================================================
def retry_on_rate_limit(client, model, messages, max_retries=5):
    """Catches RateLimitError and retries with exponential backoff."""
    for attempt in range(max_retries):
        try:
            if isinstance(client, anthropic.Anthropic):
                return client.messages.create(
                    model=model,
                    max_tokens=512,
                    messages=messages
                )
            elif isinstance(client, OpenAI):
                return client.chat.completions.create(
                    model=model,
                    max_tokens=512,
                    messages=messages
                )
            else:
                raise ValueError("Unsupported client type")
                
        except (anthropic.RateLimitError, openai.RateLimitError) as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts.")
                raise e
            wait_time = 2 ** attempt
            print(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)

# =========================================================
# 2. Token Budget Manager
# =========================================================
class BudgetExceeded(Exception):
    pass

class TokenBudgetManager:
    def __init__(self, limit: int):
        self.limit = limit
        self.used_tokens = 0

    def add_usage(self, tokens: int):
        self.used_tokens += tokens
        if self.used_tokens > self.limit:
            raise BudgetExceeded(f"Token budget of {self.limit} exceeded! Used: {self.used_tokens}")
        print(f"Budget OK: {self.used_tokens}/{self.limit} tokens used.")

# =========================================================
# 3. Compare Models (Concurrency)
# =========================================================
async def call_model_async(prompt: str, model: str) -> dict:
    start_time = time.time()
    try:
        # Determine which client to use based on model name pattern
        if "qwen" in model or "claude" in model:
            resp = await async_anthropic_client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.content[0].text
            input_tokens = resp.usage.input_tokens
            output_tokens = resp.usage.output_tokens
        else:
            resp = await async_openai_client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.choices[0].message.content
            input_tokens = resp.usage.prompt_tokens
            output_tokens = resp.usage.completion_tokens
    except Exception as e:
        text = f"Error: {e}"
        input_tokens = 0
        output_tokens = 0

    latency = (time.time() - start_time) * 1000
    
    return {
        "model": model,
        "response_text": text.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": round(latency, 2)
    }

async def compare_models_async(prompt: str, models: list[str]) -> pd.DataFrame:
    tasks = [call_model_async(prompt, model) for model in models]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)

def compare_models(prompt: str, models: list[str]) -> pd.DataFrame:
    """Calls the same prompt on multiple models concurrently and returns a DataFrame."""
    return asyncio.run(compare_models_async(prompt, models))

# =========================================================
# 4. Stream to File
# =========================================================
def stream_to_file(prompt: str, output_path: str):
    """Uses the Anthropic streaming API to write tokens to a file in real-time."""
    print(f"Streaming response to '{output_path}'...")
    with open(output_path, "w", encoding="utf-8") as f:
        with anthropic_client.messages.stream(
            model="qwen/qwen3.7-flash:free",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                f.write(text)
                f.flush() # Ensure it writes to disk immediately
    print("Streaming completed!")


# =========================================================
# Execution / Testing Area
# =========================================================
if __name__ == "__main__":
    print("\n--- Testing Ex 1: Retry ---")
    retry_on_rate_limit(
        anthropic_client, 
        model="qwen/qwen3.7-flash:free",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    print("\n--- Testing Ex 2: Budget Manager ---")
    manager = TokenBudgetManager(limit=100)
    manager.add_usage(50)
    try:
        manager.add_usage(60)
    except BudgetExceeded as e:
        print(e)

    print("\n--- Testing Ex 3: Compare Models ---")
    df = compare_models(
        prompt="Explain what a dataframe is in 1 short sentence.",
        models=["qwen/qwen3.7-flash:free", "openai/gpt-4o-mini"]
    )
    print(df.to_string())

    print("\n--- Testing Ex 4: Stream to File ---")
    stream_to_file("Write a short poem about coding.", "coding_poem.txt")
