import pytest

from docqa.config import load_settings
from docqa.loaders import load_bytes, load_file
from tests.conftest import make_pdf


def test_defaults_work_without_any_configuration():
    s = load_settings({})
    assert s.anthropic_model == "claude-opus-5-5"
    assert s.effort == "low" and s.max_tokens == 8000
    assert not s.has_api_key
    assert s.chunk_size == 900 and s.chunk_overlap == 150 and s.top_k == 4


@pytest.mark.parametrize("key", ["", "   ", "your-api-key-here", "sk-ant-..."])
def test_blank_or_placeholder_keys_do_not_count_as_a_key(key):
    assert not load_settings({"ANTHROPIC_API_KEY": key}).has_api_key


def test_real_looking_key_counts():
    assert load_settings({"ANTHROPIC_API_KEY": "sk-ant-api03-abc"}).has_api_key


def test_effort_is_only_defaulted_for_models_that_support_it():
    assert load_settings({"ANTHROPIC_MODEL": "claude-sonnet-5-5"}).effort == "low"
    assert load_settings({"ANTHROPIC_MODEL": "claude-haiku-4-5"}).effort is None
    assert load_settings({"ANTHROPIC_MODEL": "claude-haiku-4-5", "ANTHROPIC_EFFORT": "high"}).effort == "high"


def test_invalid_effort_is_rejected():
    with pytest.raises(ValueError, match="ANTHROPIC_EFFORT"):
        load_settings({"ANTHROPIC_EFFORT": "turbo"})


def test_text_and_markdown_become_one_document():
    docs = load_bytes("notes.md", b"# Title\n\nSome text.")
    assert len(docs) == 1 and docs[0].metadata == {"source": "notes.md"}


def test_pdf_gives_one_document_per_page_with_page_numbers():
    docs = load_bytes("report.pdf", make_pdf(["First page text", "Second page text"]))
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "Second page" in docs[1].page_content and docs[0].metadata["source"] == "report.pdf"


def test_blank_pages_are_skipped_but_keep_their_numbering():
    docs = load_bytes("gap.pdf", make_pdf(["Page one", "", "Page three"]))
    assert [d.metadata["page"] for d in docs] == [1, 3]


def test_pdf_without_text_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError, match="no extractable text"):
        load_bytes("scan.pdf", make_pdf(["", ""]))


@pytest.mark.parametrize("name", ["image.png", "archive.zip", "noextension"])
def test_unsupported_types_are_rejected(name):
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_bytes(name, b"data")


def test_empty_text_file_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        load_bytes("blank.txt", b"   \n ")


def test_path_components_are_stripped_from_the_source_name(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world")
    assert load_file(f)[0].metadata["source"] == "a.txt"
    assert load_bytes("../../etc/passwd.txt", b"x")[0].metadata["source"] == "passwd.txt"
