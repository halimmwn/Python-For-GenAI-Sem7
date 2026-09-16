from collections.abc import Generator, Iterable


def fast_responses(responses: list[dict]) -> list[dict]:
	"""Return responses under 500 ms, sorted by token count."""
	return sorted(
		[response for response in responses if response["latency_ms"] < 500],
		key=lambda response: response["tokens"],
	)


def conversation_stats(messages: list[dict]) -> dict:
	"""Return message counts and the average words per message."""
	total_messages = len(messages)
	user_turns = sum(message.get("role") == "user" for message in messages)
	assistant_turns = sum(
		message.get("role") == "assistant" for message in messages
	)
	total_words = sum(len(message.get("content", "").split()) for message in messages)
	average_words = total_words / total_messages if total_messages else 0.0

	return {
		"total_messages": total_messages,
		"user_turns": user_turns,
		"assistant_turns": assistant_turns,
		"avg_words_per_message": average_words,
	}


def batch_items(items: Iterable, batch_size: int) -> Generator[list, None, None]:
	"""Yield items in lists containing at most batch_size elements."""
	if batch_size < 1:
		raise ValueError("batch_size must be at least 1")

	batch = []
	for item in items:
		batch.append(item)
		if len(batch) == batch_size:
			yield batch
			batch = []

	if batch:
		yield batch


def find_fast_and_cheap_models(
	fast_models: list[str], cheap_models: list[str]
) -> set[str]:
	"""Return models that are present in both lists."""
	return set(fast_models) & set(cheap_models)


if __name__ == "__main__":
	responses = [
		{"tokens": 500, "latency_ms": 450},
		{"tokens": 200, "latency_ms": 700},
		{"tokens": 300, "latency_ms": 250},
	]
	print("Exercise 1:", fast_responses(responses))

	messages = [
		{"role": "user", "content": "Hello assistant"},
		{"role": "assistant", "content": "Hello user"},
		{"role": "user", "content": "Explain Python functions"},
	]
	print("Exercise 2:", conversation_stats(messages))

	print("Exercise 3:", list(batch_items(range(7), 3)))

	fast_models = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro"]
	cheap_models = ["gemini-1.5-pro", "gpt-4o-mini", "claude-haiku"]
	print("Exercise 4:", find_fast_and_cheap_models(fast_models, cheap_models))
