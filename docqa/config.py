"""Settings, read from environment variables (and from a local .env file if present)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "claude-opus-5-5"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Models that accept the `effort` setting. Older models (for example Haiku 4.5) reject it.
EFFORT_MODEL_PREFIXES = ("claude-opus-5", "claude-sonnet-5", "claude-fable", "claude-mythos")
PLACEHOLDER_KEYS = {"", "your-api-key-here", "your_api_key_here", "sk-ant-..."}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    collection: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embeddings_backend: str
    embedding_model: str
    anthropic_api_key: str
    anthropic_model: str
    effort: str | None
    max_tokens: int

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def demo_dir(self) -> Path:
        return self.data_dir / "demo"

    @property
    def has_api_key(self) -> bool:
        return self.anthropic_api_key.strip().lower() not in PLACEHOLDER_KEYS


def load_settings(env: dict | None = None) -> Settings:
    """Build Settings. Pass `env` to bypass the process environment and .env (used by tests)."""
    if env is None:
        load_dotenv()  # reads .env if one exists; real environment variables win over it
        env = os.environ

    def get(name: str, default: str) -> str:
        return env.get(name, default).strip() or default

    model = get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    effort = env.get("ANTHROPIC_EFFORT", "").strip().lower() or None
    if effort is None and model.startswith(EFFORT_MODEL_PREFIXES):
        effort = "low"
    if effort is not None and effort not in EFFORT_LEVELS:
        raise ValueError(f"ANTHROPIC_EFFORT must be one of {EFFORT_LEVELS}, got {effort!r}")

    return Settings(
        data_dir=Path(get("DOCQA_DATA_DIR", "data")),
        collection="documents",
        chunk_size=int(get("DOCQA_CHUNK_SIZE", "900")),
        chunk_overlap=int(get("DOCQA_CHUNK_OVERLAP", "150")),
        top_k=int(get("DOCQA_TOP_K", "4")),
        embeddings_backend=get("DOCQA_EMBEDDINGS", "fastembed").lower(),
        embedding_model=get("DOCQA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", "").strip(),
        anthropic_model=model,
        effort=effort,
        max_tokens=int(get("ANTHROPIC_MAX_TOKENS", "8000")),
    )
