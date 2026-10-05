"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- LLM (OpenAI-compatible endpoint) ---
# Set OPENAI_BASE_URL to point at any OpenAI-compatible API
# (vLLM, LiteLLM, OpenRouter, Ollama, etc.).
# Leave BLANK/empty to fall back to official OpenAI (https://api.openai.com/v1).
# Aliases LLM_BASE_URL / LLM_MODEL / MODEL_NAME are also accepted.
_OPENAI_BASE_URL_RAW = os.getenv("OPENAI_BASE_URL", "")
if not _OPENAI_BASE_URL_RAW:
    _OPENAI_BASE_URL_RAW = os.getenv("LLM_BASE_URL", "")
OPENAI_BASE_URL = _OPENAI_BASE_URL_RAW.strip().strip('"').strip("'")

_OPENAI_MODEL_RAW = os.getenv("OPENAI_MODEL", "")
if not _OPENAI_MODEL_RAW:
    _OPENAI_MODEL_RAW = os.getenv("LLM_MODEL", "")
if not _OPENAI_MODEL_RAW:
    _OPENAI_MODEL_RAW = os.getenv("MODEL_NAME", "")
OPENAI_MODEL = _OPENAI_MODEL_RAW.strip() or "gpt-4o-mini"


def get_llm_base_url(base_url: str | None = None) -> str:
    """Resolve effective base URL. Explicit arg > env. Returns "" = use OpenAI default."""
    if base_url is not None and str(base_url).strip():
        return str(base_url).strip()
    return OPENAI_BASE_URL


def get_llm_model(model: str | None = None) -> str:
    """Resolve effective chat model name. Explicit arg > env > default."""
    if model is not None and str(model).strip():
        return str(model).strip()
    return OPENAI_MODEL


def get_openai_client(base_url: str | None = None, api_key: str | None = None):
    """Build an OpenAI-compatible client.

    - If resolved base_url is blank/empty → official OpenAI endpoint
      (OpenAI() with no base_url override).
    - Else → OpenAI(base_url=..., api_key=...).

    Args:
        base_url: override endpoint. None = use OPENAI_BASE_URL env.
        api_key: override key. None = use OPENAI_API_KEY env.
    """
    from openai import OpenAI

    resolved_base = get_llm_base_url(base_url)
    resolved_key = api_key if api_key is not None else OPENAI_API_KEY
    if resolved_base:
        kwargs: dict = {"api_key": resolved_key or "sk-dummy"}
        kwargs["base_url"] = resolved_base
        return OpenAI(**kwargs)
    if resolved_key:
        return OpenAI(api_key=resolved_key)
    return OpenAI()

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
