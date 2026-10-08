from docqa.chunking import split_documents
from docqa.loaders import load_bytes
from docqa.pipeline import build_store, ingest_bytes
from tests.conftest import BANK, CATS, SPACE


def add_all(store, settings):
    for name, text in {"cats.txt": CATS, "bank.txt": BANK, "space.txt": SPACE}.items():
        ingest_bytes(store, settings, name, text.encode())


def test_search_returns_the_most_relevant_document_first(store, settings):
    add_all(store, settings)
    hits = store.search("what do cats do at night", k=3)
    assert hits[0].document.metadata["source"] == "cats.txt"
    assert hits[0].score > hits[-1].score
    assert all(0 <= h.score <= 1.0001 for h in hits)


def test_each_topic_finds_its_own_source(store, settings):
    add_all(store, settings)
    assert store.search("savings account interest", k=1)[0].document.metadata["source"] == "bank.txt"
    assert store.search("astronauts in orbit", k=1)[0].document.metadata["source"] == "space.txt"


def test_ingesting_the_same_file_twice_does_not_duplicate(store, settings):
    total, new = ingest_bytes(store, settings, "cats.txt", CATS.encode())
    assert (total, new) == (1, 1)
    total, new = ingest_bytes(store, settings, "cats.txt", CATS.encode())
    assert (total, new) == (1, 0)
    assert store.count() == 1


def test_empty_store_returns_no_hits(store):
    assert store.count() == 0
    assert store.search("anything") == []


def test_k_limits_the_number_of_results(store, settings):
    add_all(store, settings)
    assert len(store.search("cats", k=2)) == 2
    assert len(store.search("cats", k=10)) == 3  # only three chunks exist


def test_data_survives_reopening_the_store(settings):
    first = build_store(settings)
    ingest_bytes(first, settings, "cats.txt", CATS.encode())
    reopened = build_store(settings)
    assert reopened.count() == 1 and reopened.sources() == {"cats.txt": 1}


def test_sources_lists_files_with_chunk_counts(store, settings):
    add_all(store, settings)
    assert store.sources() == {"bank.txt": 1, "cats.txt": 1, "space.txt": 1}


def test_delete_source_removes_only_that_file(store, settings):
    add_all(store, settings)
    store.delete_source("cats.txt")
    assert "cats.txt" not in store.sources() and store.count() == 2


def test_reset_empties_the_store_and_it_can_be_reused(store, settings):
    add_all(store, settings)
    store.reset()
    assert store.count() == 0
    ingest_bytes(store, settings, "cats.txt", CATS.encode())
    assert store.count() == 1


def test_pdf_page_numbers_reach_the_search_results(store, settings):
    from tests.conftest import make_pdf

    ingest_bytes(store, settings, "doc.pdf", make_pdf(["Nothing about it here", CATS]))
    assert store.search("cats sleep and hunt mice", k=1)[0].document.metadata["page"] == 2
