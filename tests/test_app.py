from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from docqa.config import load_settings
from docqa.pipeline import build_store, ingest_bytes
from tests.conftest import CATS

APP = str(Path(__file__).resolve().parent.parent / "app" / "chat.py")


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    st.cache_resource.clear()  # the app caches its store for the whole process; start every test fresh
    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCQA_EMBEDDINGS", "hashing")
    return tmp_path


def test_app_starts_in_search_only_mode_with_an_empty_store(app_env):
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert any("Search-only mode" in i.value for i in app.sidebar.info)
    assert any("Nothing yet" in c.value for c in app.sidebar.caption)


def test_app_announces_full_ai_mode_when_a_key_is_set(app_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert any("Full AI mode" in s.value and "claude-opus-5-5" in s.value for s in app.sidebar.success)


def test_asking_a_question_shows_an_answer_with_sources(app_env):
    settings = load_settings()
    ingest_bytes(build_store(settings), settings, "cats.txt", CATS.encode())

    app = AppTest.from_file(APP, default_timeout=60).run()
    assert any("cats.txt" in m.value for m in app.sidebar.markdown)
    app.chat_input[0].set_value("what do cats do at night").run()

    assert not app.exception
    assert any("Showing the most relevant passages" in m.value for m in app.markdown)
    assert any("cats.txt" in m.value for m in app.markdown)
