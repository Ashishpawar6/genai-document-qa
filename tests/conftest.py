import pytest
from fpdf import FPDF

from docqa import config
from docqa.config import load_settings
from docqa.pipeline import build_store


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Tests must never read a developer's real .env or API key, or call the paid API."""
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for name in [n for n in __import__("os").environ if n.startswith(("ANTHROPIC_", "DOCQA_"))]:
        monkeypatch.delenv(name)


@pytest.fixture
def settings(tmp_path):
    return load_settings({"DOCQA_DATA_DIR": str(tmp_path), "DOCQA_EMBEDDINGS": "hashing"})


@pytest.fixture
def store(settings):
    return build_store(settings)


def make_pdf(pages: list[str]) -> bytes:
    """A small real PDF with one text page per list item (an empty string makes a blank page)."""
    pdf = FPDF()
    for text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        if text:
            pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


CATS = "Cats are small carnivorous mammals. A cat sleeps most of the day and hunts mice at night."
BANK = "A bank account holds money. Interest rates on savings accounts changed this year."
SPACE = "The rocket reached orbit after launch. Astronauts aboard the station study microgravity."
