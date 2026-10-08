"""Read PDF, text and Markdown files into LangChain Documents."""
import io
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


def load_bytes(filename: str, data: bytes) -> list[Document]:
    """Turn an uploaded file into Documents. PDFs give one Document per page (1-indexed `page`)."""
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type {suffix or '(none)'!r}; use one of {sorted(SUPPORTED)}")

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        docs = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": name, "page": number}))
        if not docs:
            raise ValueError(f"{name} contains no extractable text (a scanned PDF would need OCR)")
        return docs

    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"{name} is empty")
    return [Document(page_content=text, metadata={"source": name})]


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    return load_bytes(path.name, path.read_bytes())
