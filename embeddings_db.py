from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


def _sha256_hex(text: str) -> str:
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_embedding_key(model_name: str, text: str) -> str:
	"""Stable key for (model_name, text)."""
	h = hashlib.sha256()
	h.update(model_name.encode("utf-8"))
	h.update(b"\0")
	h.update(text.encode("utf-8"))
	return h.hexdigest()


@dataclass(frozen=True)
class EmbeddingRecord:
	key: str
	model_name: str
	text: str
	dim: int
	normalized: bool
	embedding: np.ndarray
	dataset_tag: Optional[str] = None
	prompt_id: Optional[int] = None


class EmbeddingStore:
	"""Tiny SQLite-backed store for prompt embeddings.

	- Stores float32 embeddings as BLOB.
	- Uses a SHA256 key over (model_name, text).
	"""

	def __init__(self, db_path: str):
		self.db_path = db_path
		dirname = os.path.dirname(os.path.abspath(db_path))
		os.makedirs(dirname, exist_ok=True)
		self._init_db()

	def _connect(self) -> sqlite3.Connection:
		conn = sqlite3.connect(self.db_path, timeout=30)
		conn.row_factory = sqlite3.Row
		try:
			conn.execute("PRAGMA journal_mode=WAL;")
		except sqlite3.DatabaseError:
			# Some environments/filesystems don't support WAL.
			pass
		return conn

	def _init_db(self) -> None:
		with self._connect() as conn:
			conn.execute(
				"""
				CREATE TABLE IF NOT EXISTS embeddings (
					key TEXT PRIMARY KEY,
					model_name TEXT NOT NULL,
					text TEXT NOT NULL,
					text_sha256 TEXT NOT NULL,
					dim INTEGER NOT NULL,
					normalized INTEGER NOT NULL,
					embedding BLOB NOT NULL,
					dataset_tag TEXT,
					prompt_id INTEGER,
					created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
					updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
				);
				"""
			)
			conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name);")
			conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_textsha ON embeddings(text_sha256);")
			conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_promptid ON embeddings(prompt_id);")

	@staticmethod
	def _chunks(items: list[str], chunk_size: int) -> Iterable[list[str]]:
		for i in range(0, len(items), chunk_size):
			yield items[i : i + chunk_size]

	def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
		if not keys:
			return {}

		found: dict[str, np.ndarray] = {}
		with self._connect() as conn:
			for chunk in self._chunks(keys, 900):
				placeholders = ",".join(["?"] * len(chunk))
				rows = conn.execute(
					f"SELECT key, dim, embedding FROM embeddings WHERE key IN ({placeholders})",
					chunk,
				).fetchall()
				for r in rows:
					dim = int(r["dim"])
					blob = r["embedding"]
					arr = np.frombuffer(blob, dtype=np.float32, count=dim)
					found[str(r["key"])] = arr
		return found

	def upsert_many(self, records: list[EmbeddingRecord]) -> None:
		if not records:
			return

		values = []
		for rec in records:
			arr = np.asarray(rec.embedding, dtype=np.float32).reshape(-1)
			values.append(
				(
					rec.key,
					rec.model_name,
					rec.text,
					_sha256_hex(rec.text),
					int(arr.shape[0]),
					1 if rec.normalized else 0,
					arr.tobytes(),
					rec.dataset_tag,
					rec.prompt_id,
				)
			)

		with self._connect() as conn:
			conn.executemany(
				"""
				INSERT INTO embeddings(
					key, model_name, text, text_sha256, dim, normalized, embedding, dataset_tag, prompt_id
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
				ON CONFLICT(key) DO UPDATE SET
					model_name=excluded.model_name,
					text=excluded.text,
					text_sha256=excluded.text_sha256,
					dim=excluded.dim,
					normalized=excluded.normalized,
					embedding=excluded.embedding,
					dataset_tag=excluded.dataset_tag,
					prompt_id=excluded.prompt_id,
					updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now');
				""",
				values,
			)
