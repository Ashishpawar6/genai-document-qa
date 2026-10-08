"""Embedding models: turn text into vectors so that similar meanings are close together."""
import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from docqa.config import Settings


class FastEmbedEmbeddings(Embeddings):
    """A small local sentence-embedding model (ONNX, no GPU or API key needed).

    BAAI/bge-small-en-v1.5 maps a text to a 384-number vector. It is downloaded on first use (about 65 MB).
    Documents and queries are embedded differently on purpose: the model was trained with a short
    instruction in front of search queries.
    """

    def __init__(self, model_name: str, cache_dir: str):
        self.model_name, self.cache_dir = model_name, cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding  # imported lazily: it is slow to import

            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._load().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().query_embed(text))).tolist()


class HashingEmbeddings(Embeddings):
    """A tiny, deterministic bag-of-words embedding. For tests and offline demos only.

    Each word is hashed into one of `dim` buckets and the counts are normalised, so texts that share
    words end up close together. It has no understanding of meaning (no synonyms), unlike a real model.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            vector[0], norm = 1.0, 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embeddings_backend == "hashing":
        return HashingEmbeddings()
    if settings.embeddings_backend == "fastembed":
        return FastEmbedEmbeddings(settings.embedding_model, str(settings.model_cache_dir))
    raise ValueError(f"Unknown DOCQA_EMBEDDINGS backend {settings.embeddings_backend!r}")
