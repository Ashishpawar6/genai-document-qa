"""Split documents into overlapping chunks small enough to embed and to fit in a prompt."""
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(docs: list[Document], chunk_size: int = 900, chunk_overlap: int = 150) -> list[Document]:
    """Chunks keep their parent's metadata and get `start_index` and a stable `chunk_id`.

    The id depends only on the source, page, position and text, so ingesting the same file twice
    produces the same ids (and the store can skip duplicates).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        key = f"{chunk.metadata.get('source')}|{chunk.metadata.get('page')}|{chunk.metadata['start_index']}|{chunk.page_content}"
        chunk.metadata["chunk_id"] = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return chunks
