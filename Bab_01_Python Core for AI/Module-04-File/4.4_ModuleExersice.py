import asyncio
import csv
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def save_conversation(history: list[dict], path: str) -> None:
	"""Serialize conversation history to a JSON file."""
	Path(path).write_text(
		json.dumps(history, indent=2),
		encoding="utf-8",
	)


def load_conversation(path: str) -> list[dict]:
	"""Deserialize conversation history from a JSON file."""
	return json.loads(Path(path).read_text(encoding="utf-8"))


async def compare_endpoints(
	urls: list[str],
) -> list[tuple[str, int, float]]:
	"""Fetch URLs concurrently and return status and response time."""
	async with httpx.AsyncClient() as client:
		async def fetch(url: str) -> tuple[str, int, float]:
			start = time.perf_counter()
			response = await client.get(url, timeout=10.0)
			elapsed_ms = (time.perf_counter() - start) * 1000
			return url, response.status_code, elapsed_ms

		return list(await asyncio.gather(*(fetch(url) for url in urls)))


def load_config(path: str, prefix: str = "APP") -> dict[str, Any]:
	"""Load JSON config and apply top-level environment overrides.

	A config key such as ``api_url`` is overridden by ``APP_API_URL``.
	Environment values are parsed as JSON when possible, otherwise kept as
	strings.
	"""
	config = json.loads(Path(path).read_text(encoding="utf-8"))
	for key in config:
		env_name = f"{prefix}_{key.upper()}"
		if env_name in os.environ:
			value = os.environ[env_name]
			try:
				config[key] = json.loads(value)
			except json.JSONDecodeError:
				config[key] = value
	return config


class CSVLogWriter:
	"""Append thread-safe LLM call records to a CSV file."""

	fieldnames = [
		"timestamp",
		"model",
		"input_tokens",
		"output_tokens",
		"latency_ms",
	]

	def __init__(self, path: str) -> None:
		self.path = Path(path)
		self._lock = threading.Lock()

	def log_call(
		self,
		model: str,
		input_tokens: int,
		output_tokens: int,
		latency_ms: float,
	) -> None:
		"""Append one LLM call record to the CSV file."""
		row = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"model": model,
			"input_tokens": input_tokens,
			"output_tokens": output_tokens,
			"latency_ms": latency_ms,
		}
		with self._lock:
			file_exists = self.path.exists() and self.path.stat().st_size > 0
			with self.path.open("a", newline="", encoding="utf-8") as file:
				writer = csv.DictWriter(file, fieldnames=self.fieldnames)
				if not file_exists:
					writer.writeheader()
				writer.writerow(row)


async def main() -> None:
	"""Run small demonstrations of the four exercises."""
	with tempfile.TemporaryDirectory() as directory:
		conversation_path = Path(directory) / "conversation.json"
		config_path = Path(directory) / "config.json"
		log_path = Path(directory) / "llm_calls.csv"

		history = [
			{"role": "user", "content": "What is RAG?"},
			{"role": "assistant", "content": "RAG retrieves relevant context."},
		]
		save_conversation(history, str(conversation_path))
		print("Exercise 1:", load_conversation(str(conversation_path)))

		endpoints = await compare_endpoints(
			[
				"https://jsonplaceholder.typicode.com/posts/1",
				"https://jsonplaceholder.typicode.com/posts/2",
			]
		)
		print("Exercise 2:", endpoints)

		config_path.write_text(
			json.dumps({"model": "gpt-4o", "temperature": 0.7}),
			encoding="utf-8",
		)
		os.environ["APP_TEMPERATURE"] = "0.2"
		print("Exercise 3:", load_config(str(config_path)))

		logger = CSVLogWriter(str(log_path))
		logger.log_call("gpt-4o", 100, 42, 350.5)
		print("Exercise 4:", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
	asyncio.run(main())
