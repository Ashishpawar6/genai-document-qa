import pytest

from docqa import cli
from tests.conftest import BANK, CATS, make_pdf


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCQA_EMBEDDINGS", "hashing")
    return tmp_path


@pytest.fixture
def files(tmp_path):
    (tmp_path / "cats.txt").write_text(CATS)
    (tmp_path / "bank.txt").write_text(BANK)
    (tmp_path / "doc.pdf").write_bytes(make_pdf(["Cats nap in the sun", CATS]))
    (tmp_path / "bad.png").write_bytes(b"x")
    return tmp_path


def test_ingest_status_and_ask_flow(env, files, capsys):
    assert cli.main(["ingest", str(files / "cats.txt"), str(files / "bank.txt"), str(files / "bad.png")]) == 0
    out = capsys.readouterr().out
    assert "cats.txt: 1 chunks (1 new)" in out and "Skipped" in out and "bad.png" in out
    assert "Store now holds 2 chunks" in out

    cli.main(["status"])
    out = capsys.readouterr().out
    assert "Search-only mode" in out and "cats.txt: 1 chunks" in out

    cli.main(["ask", "what do cats do at night"])
    out = capsys.readouterr().out
    assert "[1] cats.txt" in out and "similarity" in out and "ANTHROPIC_API_KEY" in out


def test_pdf_citations_include_the_page(env, files, capsys):
    cli.main(["ingest", str(files / "doc.pdf")])
    capsys.readouterr()
    cli.main(["ask", "cats hunt mice"])
    assert "doc.pdf, page 2" in capsys.readouterr().out


def test_ingesting_twice_reports_zero_new_chunks(env, files, capsys):
    cli.main(["ingest", str(files / "cats.txt")])
    cli.main(["ingest", str(files / "cats.txt")])
    assert "cats.txt: 1 chunks (0 new)" in capsys.readouterr().out


def test_reset_clears_the_store(env, files, capsys):
    cli.main(["ingest", str(files / "cats.txt")])
    cli.main(["reset"])
    capsys.readouterr()
    cli.main(["ask", "cats"])
    assert "no documents" in capsys.readouterr().out.lower()


def test_eval_command_prints_the_metrics(env, files, capsys, tmp_path):
    import json

    cli.main(["ingest", str(files / "cats.txt")])
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([{"question": "what do cats hunt", "expected_keywords": ["hunts mice"]}]))
    capsys.readouterr()
    cli.main(["eval", "--questions", str(questions), "-k", "2"])
    assert "hit@2 = 100%" in capsys.readouterr().out


def test_chat_loop_answers_then_exits(env, files, capsys, monkeypatch):
    cli.main(["ingest", str(files / "cats.txt")])
    capsys.readouterr()
    answers = iter(["what do cats do", "", "exit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert cli.main(["chat"]) == 0
    assert "[1] cats.txt" in capsys.readouterr().out


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])
