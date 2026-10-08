import json

import pytest

from docqa.evaluate import evaluate, first_correct_rank, load_questions
from docqa.pipeline import ingest_bytes
from tests.conftest import BANK, CATS, SPACE


def test_first_correct_rank_needs_every_keyword_in_one_text():
    texts = ["only cats here", "cats and mice together", "mice"]
    assert first_correct_rank(texts, ["cats", "mice"]) == 2
    assert first_correct_rank(texts, ["cats", "rocket"]) is None


def test_matching_ignores_case_and_line_breaks():
    assert first_correct_rank(["Scaled   Dot-\nProduct"], ["scaled dot-product"]) is None  # a hyphen break is not guessed at
    assert first_correct_rank(["Scaled   Dot-Product\nAttention"], ["dot-product attention"]) == 1


def test_evaluate_reports_hit_rate_and_mrr(store, settings):
    for name, text in {"cats.txt": CATS, "bank.txt": BANK, "space.txt": SPACE}.items():
        ingest_bytes(store, settings, name, text.encode())
    questions = [
        {"question": "what do cats hunt at night", "expected_keywords": ["hunts mice"]},
        {"question": "savings account interest rates", "expected_keywords": ["interest rates"]},
        {"question": "who won the football match", "expected_keywords": ["football"]},  # not in the documents
    ]
    result = evaluate(store, questions, k=3)
    assert result["questions"] == 3
    assert result["hit_at_k"] == pytest.approx(0.667, abs=0.001)
    assert result["mrr"] == pytest.approx((1 + 1 + 0) / 3, abs=0.001)
    assert [d["rank"] for d in result["details"]] == [1, 1, None]


def test_load_questions_validates_the_file(tmp_path):
    good = tmp_path / "q.json"
    good.write_text(json.dumps([{"question": "q?", "expected_keywords": ["k"]}]))
    assert len(load_questions(good)) == 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"question": "q?"}]))
    with pytest.raises(ValueError):
        load_questions(bad)


def test_the_shipped_question_file_is_valid():
    questions = load_questions("eval/attention_questions.json")
    assert len(questions) >= 10
