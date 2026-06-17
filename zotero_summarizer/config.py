"""Runtime configuration loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# Map the many spellings a user might write to a single canonical provider.
_PROVIDER_ALIASES = {
    "deepseek": "deepseek",
    "google": "google",
    "gemini": "google",
    "google-genai": "google",
    "ollama": "ollama",
    "openai": "openai",
    "openai-compatible": "openai",
    "compatible": "openai",
}

# The standard, globally-recognised env var each provider's SDK uses for its key.
# This is the "generic" naming format — set DEEPSEEK_API_KEY/OPENAI_API_KEY/etc.
# once in your shell and every tool (including this one) picks it up. Ollama
# needs no key, so it has no entry.
_PROVIDER_API_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _resolve_llm_api_key(provider: str, override: Optional[str]) -> Optional[str]:
    """Resolve the LLM API key, highest precedence first:

    1. ``override`` — a value passed explicitly (e.g. the --llm-api-key CLI flag).
    2. ``LLM_API_KEY`` — this project's generic, provider-agnostic override.
    3. The provider's standard env var (DEEPSEEK_API_KEY, OPENAI_API_KEY, ...).
    """
    if override:
        return override
    generic = os.getenv("LLM_API_KEY")
    if generic:
        return generic
    env_name = _PROVIDER_API_KEY_ENV.get(provider)
    if env_name:
        return os.getenv(env_name) or None
    return None


@dataclass
class Settings:
    # --- Zotero ---
    zotero_local: bool
    zotero_library_id: str
    zotero_library_type: str
    zotero_api_key: Optional[str]
    zotero_storage_dir: Optional[str]
    # --- LLM ---
    llm_provider: str
    llm_model: str
    llm_api_key: Optional[str]
    llm_base_url: Optional[str]
    llm_temperature: float
    # --- Behaviour ---
    max_pdf_chars: int


def _default_storage_dir() -> Optional[str]:
    """Fall back to the standard macOS/Linux Zotero storage path if present."""
    candidate = os.path.expanduser("~/Zotero/storage")
    return candidate if os.path.isdir(candidate) else None


def load_settings(
    *,
    llm_api_key: Optional[str] = None,
    zotero_api_key: Optional[str] = None,
) -> Settings:
    """Build Settings from the environment.

    ``llm_api_key`` / ``zotero_api_key`` are optional explicit overrides (the CLI
    passes its --llm-api-key / --zotero-api-key flags here); when given, they win
    over anything in the environment.
    """
    raw_provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    provider = _PROVIDER_ALIASES.get(raw_provider, raw_provider)
    return Settings(
        zotero_local=_bool("ZOTERO_LOCAL", True),
        zotero_library_id=os.getenv("ZOTERO_LIBRARY_ID", "0"),
        zotero_library_type=os.getenv("ZOTERO_LIBRARY_TYPE", "user"),
        zotero_api_key=zotero_api_key or os.getenv("ZOTERO_API_KEY") or None,
        zotero_storage_dir=os.getenv("ZOTERO_STORAGE_DIR") or _default_storage_dir(),
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", "deepseek-chat"),
        llm_api_key=_resolve_llm_api_key(provider, llm_api_key),
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_pdf_chars=int(os.getenv("MAX_PDF_CHARS", "48000")),
    )


def build_llm(settings: Settings):
    """Construct a LangChain chat model. Defaults to DeepSeek; supports any
    OpenAI-compatible endpoint via LLM_PROVIDER=openai + LLM_BASE_URL.

    The API key in ``settings.llm_api_key`` is already resolved (CLI flag >
    LLM_API_KEY > the provider's standard env var); see ``load_settings``."""
    provider = settings.llm_provider
    common = {"temperature": settings.llm_temperature}

    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            **common,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.llm_api_key,
            **common,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            # 127.0.0.1, not localhost: localhost can resolve to IPv6 (::1),
            # which on some machines is a different (e.g. Docker) listener.
            base_url=settings.llm_base_url or "http://127.0.0.1:11434",
            # Native Ollama structured output uses a real JSON schema (reliable).
            # Disable qwen3's <think> step: faster and keeps the JSON clean.
            reasoning=False,
            **common,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            **common,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        f"Use 'deepseek', 'google', 'ollama', or 'openai'."
    )
