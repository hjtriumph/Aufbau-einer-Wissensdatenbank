from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

def _to_int(x: str | None, default: int) -> int:
    try:
        return int(x) if x is not None else default
    except ValueError:
        return default

def _to_float(x: str | None, default: float) -> float:
    try:
        return float(x) if x is not None else default
    except ValueError:
        return default

def _project_root() -> Path:
    # 假设 config.py 在 src/ 下：src/config.py -> root 是上一级
    return Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    # LLM (DeepSeek OpenAI-compatible)
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str

    # LLM runtime params (for reproducibility)
    temperature: float
    max_tokens: int | None
    request_timeout: float | None
    max_retries: int

    # Embeddings
    embedding_model: str

    # Paths / storage
    docs_dir: str
    chroma_dir: str
    collection_name: str

    # Retrieval params
    k: int
    fetch_k: int
    search_type: str  # "similarity" or "mmr"

    chunk_size: int
    chunk_overlap: int

    @staticmethod
    def from_env() -> "Settings":
        root = _project_root()

        docs_dir = os.getenv("DOCS_DIR", str(root / "data" / "docs"))
        chroma_dir = os.getenv("CHROMA_DIR", str(root / "storage" / "chroma"))

        # normalize to absolute paths
        docs_dir = str(Path(docs_dir).expanduser().resolve())
        chroma_dir = str(Path(chroma_dir).expanduser().resolve())

        max_tokens_env = os.getenv("MAX_TOKENS", "").strip()
        max_tokens = int(max_tokens_env) if max_tokens_env else None

        timeout_env = os.getenv("REQUEST_TIMEOUT", "").strip()
        request_timeout = float(timeout_env) if timeout_env else None

        return Settings(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),

            temperature=_to_float(os.getenv("TEMPERATURE"), 0.2),
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            max_retries=_to_int(os.getenv("MAX_RETRIES"), 2),

            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),

            docs_dir=docs_dir,
            chroma_dir=chroma_dir,
            collection_name=os.getenv("COLLECTION_NAME", "building_tech_kb"),

            k=_to_int(os.getenv("K"), 6),
            fetch_k=_to_int(os.getenv("FETCH_K"), 20),
            search_type=os.getenv("SEARCH_TYPE", "mmr"),

            chunk_size=_to_int(os.getenv("CHUNK_SIZE"), 1000),
            chunk_overlap=_to_int(os.getenv("CHUNK_OVERLAP"), 150),
        )

settings = Settings.from_env()
