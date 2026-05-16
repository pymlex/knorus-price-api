from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, key: str) -> np.ndarray | None:
        row = self.conn.execute(
            "SELECT dim, vector FROM embeddings WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        dim, blob = row
        return np.frombuffer(blob, dtype=np.float32, count=dim).copy()

    def set(self, key: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings(key, dim, vector) VALUES(?, ?, ?)",
            (key, int(vector.shape[0]), vector.tobytes()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


@dataclass
class TextEmbedder:
    model_id: str
    cache_path: Path
    max_length: int
    batch_size: int

    def __post_init__(self) -> None:
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=_dtype(),
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.model.config.use_cache = False
        self.cache = EmbeddingCache(self.cache_path)

    def close(self) -> None:
        self.cache.close()

    def _mean_pool(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
        summed = (hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.inference_mode()
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        pooled = self._mean_pool(hidden, encoded["attention_mask"])
        return pooled.float().cpu().numpy().astype(np.float32)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        normalized = [text.strip() for text in texts]
        cached_vectors: list[np.ndarray | None] = []
        missing_texts: list[str] = []
        missing_keys: list[str] = []

        for text in normalized:
            if not text:
                cached_vectors.append(None)
                continue
            key = _hash_text(text)
            cached = self.cache.get(key)
            if cached is None:
                cached_vectors.append(None)
                missing_texts.append(text)
                missing_keys.append(key)
            else:
                cached_vectors.append(cached)

        for start in range(0, len(missing_texts), self.batch_size):
            batch = missing_texts[start:start + self.batch_size]
            batch_keys = missing_keys[start:start + self.batch_size]
            vectors = self._embed_batch(batch)
            for key, vector in zip(batch_keys, vectors):
                self.cache.set(key, vector)

        output: list[np.ndarray] = []
        dim = None
        for text, cached in zip(normalized, cached_vectors):
            if cached is not None:
                dim = cached.shape[0]
                output.append(cached)
                continue
            if not text:
                if dim is None:
                    probe = self.cache.conn.execute("SELECT dim FROM embeddings LIMIT 1").fetchone()
                    if probe is None:
                        raise RuntimeError("Embedding cache is empty.")
                    dim = int(probe[0])
                output.append(np.zeros(dim, dtype=np.float32))
                continue
            key = _hash_text(text)
            refreshed = self.cache.get(key)
            if refreshed is None:
                raise RuntimeError("Embedding cache miss after refresh.")
            dim = refreshed.shape[0]
            output.append(refreshed)

        return np.vstack(output).astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_group_mean(self, labels: list[str]) -> np.ndarray:
        if not labels:
            probe = self.cache.conn.execute("SELECT dim FROM embeddings LIMIT 1").fetchone()
            if probe is None:
                raise RuntimeError("Embedding cache is empty.")
            return np.zeros(int(probe[0]), dtype=np.float32)
        vectors = self.embed_texts(labels)
        return vectors.mean(axis=0).astype(np.float32)
