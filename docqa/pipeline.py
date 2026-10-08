"""Glue: build the store/QA objects from settings and ingest files."""
from pathlib import Path

from docqa.chunking import split_documents
from docqa.config import Settings
from docqa.embeddings import get_embeddings
from docqa.llm import get_llm
from docqa.loaders import load_bytes
from docqa.qa import DocumentQA
from docqa.store import DocumentStore

DEMO_URL = "https://arxiv.org/pdf/1706.03762"  # "Attention Is All You Need" (Vaswani et al., 2017)
DEMO_NAME = "attention-is-all-you-need.pdf"
MAX_DEMO_BYTES = 20 * 1024 * 1024


def build_store(settings: Settings) -> DocumentStore:
    return DocumentStore(get_embeddings(settings), settings.chroma_dir, settings.collection)


def build_qa(settings: Settings, store: DocumentStore | None = None) -> DocumentQA:
    return DocumentQA(store or build_store(settings), get_llm(settings), settings.top_k)


def ingest_bytes(store: DocumentStore, settings: Settings, filename: str, data: bytes) -> tuple[int, int]:
    """Ingest one file. Returns (chunks in the file, chunks that were new)."""
    chunks = split_documents(load_bytes(filename, data), settings.chunk_size, settings.chunk_overlap)
    return len(chunks), store.add(chunks)


def ingest_path(store: DocumentStore, settings: Settings, path: str | Path) -> tuple[int, int]:
    path = Path(path)
    return ingest_bytes(store, settings, path.name, path.read_bytes())


def download_demo(settings: Settings) -> Path:
    """Download the demo paper from arXiv (not stored in the repo)."""
    import urllib.request

    target = settings.demo_dir / DEMO_NAME
    if target.exists():
        return target
    settings.demo_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DEMO_URL, timeout=60) as response:  # noqa: S310 - fixed https URL
        data = response.read(MAX_DEMO_BYTES + 1)
    if len(data) > MAX_DEMO_BYTES or not data.startswith(b"%PDF"):
        raise RuntimeError("The downloaded demo file is not a valid PDF")
    target.write_bytes(data)
    return target
