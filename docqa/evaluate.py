"""Measure retrieval quality: does the right passage come back for each question?

Each test question lists keywords that must ALL appear in one retrieved chunk for it to count as a hit.
Reported metrics:
  hit@k  share of questions where a correct chunk is in the top k results
  MRR    mean reciprocal rank: average of 1/rank of the first correct chunk (0 if none)
"""
import json
import re
from pathlib import Path

from docqa.store import DocumentStore


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def load_questions(path: str | Path) -> list[dict]:
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    for q in questions:
        if not q.get("question") or not q.get("expected_keywords"):
            raise ValueError(f"Each item needs 'question' and 'expected_keywords': {q}")
    return questions


def first_correct_rank(texts: list[str], keywords: list[str]) -> int | None:
    """1-based rank of the first text that contains every keyword, or None."""
    needed = [_normalise(k) for k in keywords]
    for rank, text in enumerate(texts, start=1):
        haystack = _normalise(text)
        if all(k in haystack for k in needed):
            return rank
    return None


def evaluate(store: DocumentStore, questions: list[dict], k: int = 4) -> dict:
    details, reciprocal = [], []
    for q in questions:
        hits = store.search(q["question"], k=k)
        rank = first_correct_rank([h.document.page_content for h in hits], q["expected_keywords"])
        reciprocal.append(1 / rank if rank else 0.0)
        details.append({"question": q["question"], "rank": rank})
    n = len(questions)
    return {
        "k": k,
        "questions": n,
        "hit_at_k": round(sum(d["rank"] is not None for d in details) / n, 3) if n else 0.0,
        "mrr": round(sum(reciprocal) / n, 3) if n else 0.0,
        "details": details,
    }
