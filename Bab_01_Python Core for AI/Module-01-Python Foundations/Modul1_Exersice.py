import functools
import time


def token_cost(tokens: int, model: str) -> float:
	"""Return the estimated cost for the given token count and model."""
	costs_per_1k_tokens = {
		"gpt-4o": 0.005,
		"claude-sonnet-4-5": 0.003,
		"gemini-1.5-pro": 0.00125,
	}

	if model not in costs_per_1k_tokens:
		raise ValueError(f"Unknown model: {model}")

	return tokens / 1000 * costs_per_1k_tokens[model]


def retry(max_attempts: int, delay: float):
	"""Retry a function up to max_attempts times after an exception."""
	def decorator(function):
		@functools.wraps(function)
		def wrapper(*args, **kwargs):
			for attempt in range(max_attempts):
				try:
					return function(*args, **kwargs)
				except Exception:
					if attempt == max_attempts - 1:
						raise
					time.sleep(delay)

		return wrapper

	return decorator


def temperature_label(t: float) -> str:
	"""Return a label for a temperature from 0.0 through 1.0."""
	if not 0.0 <= t <= 1.0:
		raise ValueError("temperature must be between 0.0 and 1.0")

	if t < 0.3:
		return "precise"
	if t < 0.7:
		return "balanced"
	return "creative"


def parse_token_cost(text: str) -> tuple[int, float]:
	"""Extract the token count and cost from the exercise text."""
	token_part, cost_part = text.split(",")
	token_count = int(token_part.strip().split()[0])
	cost = float(cost_part.strip().split()[0])
	return token_count, cost


def test_retry() -> str:
	"""Test retry with a function that fails on its first two calls."""
	state = {"attempts": 0}

	@retry(max_attempts=3, delay=0.01)
	def flaky_function() -> str:
		state["attempts"] += 1
		if state["attempts"] < 3:
			raise RuntimeError("Temporary failure")
		return "Success"

	return flaky_function()


if __name__ == "__main__":
	print("Exercise 1:", token_cost(128000, "gpt-4o"))
	print("Exercise 2:", test_retry())
	print("Exercise 3:", temperature_label(0.5))
	print("Exercise 4:", parse_token_cost("128000 tokens, 0.005 USD per 1K"))
