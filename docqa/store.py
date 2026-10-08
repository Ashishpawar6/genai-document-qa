"""A vector store (ChromaDB) holding document chunks and their embeddings."""
from dataclasses import dataclass
from pathlib import Path

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass
class Hit:
    document: Document
    score: float  # cosine similarity, higher is closer (about 0 to 1)


class DocumentStore:
    def __init__(self, embeddings: Embeddings, persist_dir: Path, collection: str = "documents"):
        self._embeddings = embeddings
        self._persist_dir = Path(persist_dir)
        self._collection = collection
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._db = self._open()

    def _open(self) -> Chroma:
        return Chroma(
            collection_name=self._collection,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
            collection_metadata={"hnsw:space": "cosine"},  # compare vectors by angle (cosine similarity)
            client_settings=ChromaSettings(
                anonymized_telemetry=False, is_persistent=True, persist_directory=str(self._persist_dir)
            ),
        )

    def add(self, chunks: list[Document]) -> int:
        """Add chunks, skipping any already stored (matched by chunk_id). Returns how many were new."""
        if not chunks:
            return 0
        ids = [c.metadata["chunk_id"] for c in chunks]
        existing = set(self._db.get(ids=ids)["ids"])
        new = [(i, c) for i, c in zip(ids, chunks) if i not in existing]
        if new:
            self._db.add_documents([c for _, c in new], ids=[i for i, _ in new])
        return len(new)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if self.count() == 0:
            return []
        results = self._db.similarity_search_with_score(query, k=k)
        hits = [Hit(document=doc, score=round(1.0 - distance, 4)) for doc, distance in results]
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def count(self) -> int:
        return len(self._db.get(include=[])["ids"])

    def sources(self) -> dict[str, int]:
        """Map each ingested file name to its number of chunks."""
        counts: dict[str, int] = {}
        for meta in self._db.get(include=["metadatas"])["metadatas"]:
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
        return dict(sorted(counts.items()))

    def delete_source(self, source: str) -> None:
        self._db.delete(where={"source": source})

    def reset(self) -> None:
        self._db.delete_collection()
        self._db = self._open()
