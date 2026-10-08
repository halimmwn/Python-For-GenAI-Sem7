import os
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3
import hashlib
import json
import re
from collections import Counter
import math

load_dotenv()

openai_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# =========================================================
# Common Data Structures
# =========================================================
@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int

# =========================================================
# Exercise 1: DuplicateDetector
# =========================================================
class DuplicateDetector:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def find_duplicates(self, documents: List[str]) -> List[Tuple[str, str, float]]:
        if not documents:
            return []
        
        # Uses embed_with_cache from Exercise 4
        embeddings = embed_with_cache(documents)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        duplicates = []
        n = len(documents)
        for i in range(n):
            for j in range(i + 1, n):
                score = similarity_matrix[i, j]
                if score >= self.threshold:
                    duplicates.append((documents[i], documents[j], float(score)))
                    
        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates

# =========================================================
# Exercise 2: HybridSearch
# =========================================================
class HybridSearch:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        # Uses VectorStore from Exercise 3
        self.vector_store = VectorStore()
        
    def add_documents(self, documents: List[Document]):
        self.vector_store.add_documents(documents)
        
    def _compute_bm25_scores(self, query: str, documents: List[Document]) -> List[float]:
        def tokenize(text: str):
            return re.findall(r'\w+', text.lower())
            
        doc_tokens = [tokenize(d.text) for d in documents]
        avgdl = sum(len(dt) for dt in doc_tokens) / len(doc_tokens) if doc_tokens else 1
        N = len(documents)
        
        query_tokens = tokenize(query)
        
        k1 = 1.5
        b = 0.75
        
        scores = []
        for dt in doc_tokens:
            score = 0.0
            dt_len = len(dt)
            dt_counts = Counter(dt)
            for q in query_tokens:
                nq = sum(1 for doc in doc_tokens if q in doc)
                idf = math.log((N - nq + 0.5) / (nq + 0.5) + 1)
                
                tf = dt_counts.get(q, 0)
                num = tf * (k1 + 1)
                den = tf + k1 * (1 - b + b * dt_len / avgdl)
                score += idf * (num / den)
            scores.append(score)
            
        max_score = max(scores) if scores and max(scores) > 0 else 1.0
        return [s / max_score for s in scores]

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if not self.vector_store._documents:
            return []
            
        # Semantic Scores 
        q_vec = embed_with_cache([query], model=self.vector_store.embed_model)[0]
        semantic_scores = self.vector_store._matrix @ q_vec
        
        # Keyword Scores 
        keyword_scores = self._compute_bm25_scores(query, self.vector_store._documents)
        
        # Blended Scores
        final_scores = []
        for i, doc in enumerate(self.vector_store._documents):
            semantic = semantic_scores[i]
            keyword = keyword_scores[i]
            final = self.alpha * semantic + (1 - self.alpha) * keyword
            final_scores.append(final)
            
        final_scores = np.array(final_scores)
        k = min(k, len(self.vector_store._documents))
        top_idx = np.argsort(final_scores)[::-1][:k]
        
        return [
            SearchResult(
                document=self.vector_store._documents[int(i)], 
                score=float(final_scores[i]), 
                rank=rank+1
            )
            for rank, i in enumerate(top_idx)
        ]

# =========================================================
# Exercise 3: Extended VectorStore
# =========================================================
class VectorStore:
    def __init__(self, embed_model: str = "text-embedding-3-small"):
        self.embed_model = embed_model
        self._documents: List[Document] = []
        self._matrix: Optional[np.ndarray] = None

    def _rebuild_matrix(self):
        if not self._documents:
            self._matrix = None
        else:
            self._matrix = np.array([d.embedding for d in self._documents], dtype=np.float32)

    def add_documents(self, documents: List[Document]) -> None:
        texts = [d.text for d in documents]
        # Uses embed_with_cache from Exercise 4
        normed = embed_with_cache(texts, model=self.embed_model)
        
        for doc, vec in zip(documents, normed):
            doc.embedding = vec
            self._documents.append(doc)
            
        self._rebuild_matrix()

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if self._matrix is None or len(self._documents) == 0:
            return []
            
        q_vec = embed_with_cache([query], model=self.embed_model)[0]
        scores = self._matrix @ q_vec
        
        k = min(k, len(self._documents))
        top_idx = np.argsort(scores)[::-1][:k]
        
        return [
            SearchResult(document=self._documents[int(i)], score=float(scores[i]), rank=rank+1)
            for rank, i in enumerate(top_idx)
        ]

    def delete(self, doc_id: str) -> bool:
        for i, doc in enumerate(self._documents):
            if doc.id == doc_id:
                del self._documents[i]
                self._rebuild_matrix()
                return True
        return False

    def update(self, doc_id: str, new_text: str) -> bool:
        for i, doc in enumerate(self._documents):
            if doc.id == doc_id:
                doc.text = new_text
                doc.embedding = embed_with_cache([new_text], model=self.embed_model)[0]
                self._rebuild_matrix()
                return True
        return False

# =========================================================
# Exercise 4: embed_with_cache
# =========================================================
DB_FILE = "embeddings_cache.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            hash_key TEXT PRIMARY KEY,
            embedding TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_hash(text: str, model: str) -> str:
    return hashlib.sha256((text + model).encode('utf-8')).hexdigest()

def embed_with_cache(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Wrapper around OpenAI embeddings API that stores results in local SQLite."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    results = []
    missing_texts = []
    missing_indices = []
    
    for i, text in enumerate(texts):
        h = get_hash(text, model)
        c.execute('SELECT embedding FROM embeddings WHERE hash_key = ?', (h,))
        row = c.fetchone()
        if row:
            embedding = json.loads(row[0])
            results.append((i, embedding))
        else:
            missing_texts.append(text)
            missing_indices.append(i)
            
    if missing_texts:
        resp = openai_client.embeddings.create(input=missing_texts, model=model)
        for i, api_data in enumerate(sorted(resp.data, key=lambda x: x.index)):
            idx = missing_indices[i]
            embedding = api_data.embedding
            results.append((idx, embedding))
            
            h = get_hash(missing_texts[i], model)
            c.execute('INSERT OR IGNORE INTO embeddings (hash_key, embedding) VALUES (?, ?)', 
                      (h, json.dumps(embedding)))
            
        conn.commit()
    conn.close()
    
    results.sort(key=lambda x: x[0])
    vecs = np.array([r[1] for r in results], dtype=np.float32)
    
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


# =========================================================
# Testing all Exercises
# =========================================================
if __name__ == "__main__":
    
    print("=== Exercise 1: DuplicateDetector ===")
    db_path = Path(__file__).with_name("demam_berdarah.db")
    
    # Membuat/koneksi ke db dan membuat tabel jika belum ada
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT
        )
    ''')
    
    # Memasukkan teks ke dalam db jika masih kosong
    c.execute('SELECT COUNT(*) FROM documents')
    if c.fetchone()[0] == 0:
        dbd_text = (
            "Demam Berdarah Dengue (DBD) adalah penyakit yang disebabkan oleh infeksi virus "
            "dengue dan ditularkan terutama melalui gigitan nyamuk Aedes aegypti dan Aedes "
            "albopictus. Penyakit ini dapat menyebabkan demam tinggi dan pada kondisi berat "
            "dapat menyebabkan perdarahan serta penurunan tekanan darah akibat kebocoran plasma."
        )
        # Memasukkan teks dua kali agar DuplicateDetector dapat menemukan duplikat
        c.execute('INSERT INTO documents (content) VALUES (?)', (dbd_text,))
        c.execute('INSERT INTO documents (content) VALUES (?)', (dbd_text,))
        conn.commit()

    # Load corpus dari db
    c.execute('SELECT content FROM documents')
    corpus = [row[0] for row in c.fetchall()]
    conn.close()

    print(f"Loaded {len(corpus)} documents from {db_path.name}")

    detector = DuplicateDetector(threshold=0.98)
    dups = detector.find_duplicates(corpus)
    print(f"Found {len(dups)} duplicate pairs.")
    for d1, d2, score in dups:
        print(f"  Score {score:.4f}: '{d1}' | '{d2}'")
        
    print("\n=== Exercise 2: HybridSearch ===")
    hybrid = HybridSearch(alpha=0.5) # 50% semantic, 50% keyword
    hybrid.add_documents([
        Document("h1", "The quick brown fox jumps over the lazy dog."),
        Document("h2", "A fast fox leaps over a sleepy hound."),
        Document("h3", "Dogs are very loyal pets.")
    ])
    
    query = "fast fox"
    print(f"Query: '{query}'")
    results = hybrid.search(query, k=2)
    for r in results:
        print(f"  [{r.rank}] Blended Score: {r.score:.4f} | {r.document.text}")
        
    print("\n=== Exercise 3: Extended VectorStore (Delete & Update) ===")
    store = VectorStore()
    docs = [
        Document("d1", "Machine learning is fascinating."),
        Document("d2", "I love artificial intelligence."),
        Document("d3", "Python is a great programming language.")
    ]
    store.add_documents(docs)
    print(f"Store size: {len(store._documents)}")
    
    print("-> Deleting 'd2'...")
    store.delete("d2")
    print(f"Store size after delete: {len(store._documents)}")
    
    print("-> Updating 'd1'...")
    store.update("d1", "Machine learning and deep learning are fascinating.")
    print(f"Updated 'd1' new text: {store._documents[0].text}")
    
    print("\n=== Exercise 4: embed_with_cache ===")
    test_texts = ["Caching is awesome.", "OpenAI embeddings cost money."]
    print("-> First run (should call API):")
    v1 = embed_with_cache(test_texts)
    print("-> Second run (should load from DB cache):")
    v2 = embed_with_cache(test_texts)
    print(f"Vectors match? {np.allclose(v1, v2)}")
