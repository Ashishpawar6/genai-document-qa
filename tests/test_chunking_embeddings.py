import math

import pytest
from langchain_core.documents import Document

from docqa.chunking import split_documents
from docqa.embeddings import HashingEmbeddings, get_embeddings
from docqa.config import load_settings


def doc(text, **meta):
    return Document(page_content=text, metadata={"source": "a.txt", **meta})


LONG = " ".join(f"word{i}" for i in range(600))


def test_chunks_respect_the_size_limit_and_overlap():
    chunks = split_documents([doc(LONG)], chunk_size=300, chunk_overlap=60)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 300 for c in chunks)
    first_tail = chunks[0].page_content.split()[-3:]
    assert all(word in chunks[1].page_content for word in first_tail)  # neighbouring chunks share text


def test_chunks_keep_parent_metadata_and_get_positions():
    chunks = split_documents([doc(LONG, page=7)], chunk_size=300, chunk_overlap=60)
    assert all(c.metadata["source"] == "a.txt" and c.metadata["page"] == 7 for c in chunks)
    assert chunks[0].metadata["start_index"] == 0
    assert chunks[1].metadata["start_index"] > 0


def test_chunk_ids_are_unique_and_stable_across_runs():
    a = split_documents([doc(LONG)], 300, 60)
    b = split_documents([doc(LONG)], 300, 60)
    assert [c.metadata["chunk_id"] for c in a] == [c.metadata["chunk_id"] for c in b]
    assert len({c.metadata["chunk_id"] for c in a}) == len(a)


def test_same_text_on_different_pages_gets_different_ids():
    chunks = split_documents([doc("same text", page=1), doc("same text", page=2)])
    assert chunks[0].metadata["chunk_id"] != chunks[1].metadata["chunk_id"]


def test_short_document_is_one_chunk_and_empty_input_gives_none():
    assert len(split_documents([doc("tiny")])) == 1
    assert split_documents([]) == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_documents([doc("x")], chunk_size=100, chunk_overlap=100)


def test_hashing_embeddings_are_deterministic_unit_vectors():
    e = HashingEmbeddings()
    v1, v2 = e.embed_query("the quick brown fox"), e.embed_query("the quick brown fox")
    assert v1 == v2 and len(v1) == 256
    assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-9)
    assert math.isclose(sum(x * x for x in e.embed_query("")), 1.0)  # even empty text gives a valid vector


def test_texts_sharing_words_are_closer_than_unrelated_texts():
    e = HashingEmbeddings()
    cos = lambda a, b: sum(x * y for x, y in zip(a, b))
    q = e.embed_query("cats sleep all day")
    assert cos(q, e.embed_query("a cat sleeps all day long")) > cos(q, e.embed_query("rocket launch orbit"))


def test_backend_selection_and_unknown_backend():
    assert isinstance(get_embeddings(load_settings({"DOCQA_EMBEDDINGS": "hashing"})), HashingEmbeddings)
    with pytest.raises(ValueError, match="Unknown"):
        get_embeddings(load_settings({"DOCQA_EMBEDDINGS": "magic"}))
