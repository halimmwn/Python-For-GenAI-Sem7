import functools
import string
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from llm_core import ConversationHistory, LLMConfig


class RateLimiter:
	"""Limit calls to max_calls within a rolling time period."""

	def __init__(self, max_calls: int, period_seconds: float = 60.0) -> None:
		if max_calls < 1:
			raise ValueError("max_calls must be at least 1")
		if period_seconds <= 0:
			raise ValueError("period_seconds must be greater than 0")

		self.max_calls = max_calls
		self.period_seconds = period_seconds
		self._calls: deque[float] = deque()

	def check_and_wait(self) -> None:
		"""Wait until another call is allowed."""
		while True:
			now = time.monotonic()
			while self._calls and now - self._calls[0] >= self.period_seconds:
				self._calls.popleft()

			if len(self._calls) < self.max_calls:
				self._calls.append(now)
				return

			time.sleep(self.period_seconds - (now - self._calls[0]))


@dataclass
class PromptTemplate:
	template: str

	def render(self, **kwargs: Any) -> str:
		"""Render the template after validating all placeholders."""
		fields = {
			field_name.split(".")[0].split("[")[0]
			for _, field_name, _, _ in string.Formatter().parse(self.template)
			if field_name is not None
		}
		missing_fields = fields - kwargs.keys()
		if missing_fields:
			missing = ", ".join(sorted(missing_fields))
			raise ValueError(f"Missing template values: {missing}")
		return self.template.format_map(kwargs)


def retry(max_attempts: int, delay: float = 0.1):
	"""Retry a function up to max_attempts times after an exception."""
	if max_attempts < 1:
		raise ValueError("max_attempts must be at least 1")

	def decorator(function: Callable):
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


def test_retry() -> str:
	attempts = {"count": 0}

	@retry(max_attempts=3, delay=0.1)
	def flaky_function() -> str:
		attempts["count"] += 1
		if attempts["count"] < 3:
			raise RuntimeError("Temporary failure")
		return "Success"

	return flaky_function()


def test_rate_limiter() -> list[int]:
	limiter = RateLimiter(max_calls=2, period_seconds=0.1)
	call_numbers = []
	for call_number in range(1, 6):
		limiter.check_and_wait()
		call_numbers.append(call_number)
	return call_numbers


if __name__ == "__main__":
	print("Exercise 1:", test_rate_limiter())

	prompt = PromptTemplate("Answer {question} using a {tone} tone.")
	print("Exercise 2:", prompt.render(question="What is Python?", tone="concise"))

	print("Exercise 3:", test_retry())

	history = ConversationHistory(max_turns=2, system_prompt="Be concise.")
	history.add("user", "Hello")
	config = LLMConfig(model="claude-sonnet-4-5")
	print("Exercise 4:", history, config)
