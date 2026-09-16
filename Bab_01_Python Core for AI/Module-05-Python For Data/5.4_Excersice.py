import hashlib
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def create_benchmark_csv(path: Path) -> None:
	"""Create benchmark data with 20 rows and four models."""
	models = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro", "llama-3.1-70b"]
	tasks = ["reasoning", "summarization", "coding", "qa", "classification"]
	rows = [
		{
			"model": models[index % len(models)],
			"task": tasks[index % len(tasks)],
			"score": round(0.70 + (index % 10) * 0.025, 3),
			"latency_ms": 250 + index * 18,
		}
		for index in range(20)
	]
	pd.DataFrame(rows).to_csv(path, index=False)


def analyse_benchmarks(path: Path) -> tuple[pd.Series, pd.DataFrame, float]:
	"""Return mean scores, best task per model, and score-latency correlation."""
	benchmark = pd.read_csv(path)
	mean_score = benchmark.groupby("model")["score"].mean()
	best_rows = benchmark.loc[benchmark.groupby("model")["score"].idxmax()]
	best_tasks = best_rows[["model", "task", "score"]].sort_values("model")
	correlation = benchmark["score"].corr(benchmark["latency_ms"])
	return mean_score, best_tasks, correlation


def normalise_embeddings(matrix: np.ndarray) -> np.ndarray:
	"""L2-normalise each row of an embedding matrix."""
	norms = np.linalg.norm(matrix, axis=1, keepdims=True)
	return matrix / np.where(norms == 0, 1, norms)


def text_folder_stats(folder: Path) -> pd.DataFrame:
	"""Return sorted character, word, and sentence counts for text files."""
	rows = []
	for path in sorted(folder.glob("*.txt")):
		text = path.read_text(encoding="utf-8")
		words = text.split()
		sentences = [sentence for sentence in re.split(r"[.!?]+", text) if sentence.strip()]
		rows.append(
			{
				"filename": path.name,
				"char_count": len(text),
				"word_count": len(words),
				"sentence_count": len(sentences),
			}
		)
	return pd.DataFrame(
		rows,
		columns=["filename", "char_count", "word_count", "sentence_count"],
	).sort_values("word_count", ascending=False, ignore_index=True)


def mock_embedding(text: str, dimensions: int = 8) -> np.ndarray:
	"""Create a deterministic embedding from a text hash."""
	digest = hashlib.sha256(text.encode("utf-8")).digest()
	values = [digest[index] / 255 for index in range(dimensions)]
	return np.array(values, dtype=float)


def pairwise_cosine_similarity(texts: list[str]) -> tuple[np.ndarray, tuple[int, int]]:
	"""Build a cosine similarity matrix and return its highest pair."""
	embeddings = normalise_embeddings(np.array([mock_embedding(text) for text in texts]))
	similarities = embeddings @ embeddings.T
	upper_triangle = np.triu(similarities, k=1)
	first, second = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)
	return similarities, (int(first), int(second))


if __name__ == "__main__":
	with tempfile.TemporaryDirectory() as directory:
		workspace = Path(directory)

		benchmark_path = workspace / "benchmark.csv"
		create_benchmark_csv(benchmark_path)
		mean_score, best_tasks, correlation = analyse_benchmarks(benchmark_path)
		print("Exercise 1 - mean score:\n", mean_score)
		print("Exercise 1 - best task per model:\n", best_tasks)
		print(f"Exercise 1 - score/latency correlation: {correlation:.3f}")

		embeddings = np.array([[3.0, 4.0], [1.0, 2.0], [0.0, 5.0]])
		normalised = normalise_embeddings(embeddings)
		print("Exercise 2 - row norms:", np.linalg.norm(normalised, axis=1))

		text_folder = workspace / "documents"
		text_folder.mkdir()
		(text_folder / "short.txt").write_text("One short document.", encoding="utf-8")
		(text_folder / "long.txt").write_text(
			"This is a longer document. It has two sentences.",
			encoding="utf-8",
		)
		print("Exercise 3:\n", text_folder_stats(text_folder))

		texts = [
			"Python is useful for data analysis.",
			"Python helps with machine learning.",
			"A sunny day is good for walking.",
			"Data analysis uses useful tools.",
			"Machine learning uses data.",
		]
		similarities, pair = pairwise_cosine_similarity(texts)
		print("Exercise 4 - similarity matrix:\n", similarities)
		print("Exercise 4 - highest pair:", pair, (texts[pair[0]], texts[pair[1]]))
