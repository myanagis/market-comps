# market_comps/config.py
"""
Configuration and settings loaded from environment variables / .env file,
or from Streamlit Cloud secrets when deployed on Streamlit Community Cloud.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)


def _get_secret(key: str, default: str = "") -> str:
    """
    Read a config value — checks in priority order:
      1. Environment variable (covers .env via load_dotenv above)
      2. Streamlit secrets (st.secrets) — available on Streamlit Community Cloud
      3. Provided default
    """
    val = os.environ.get(key)
    if val:
        return val
    try:
        import streamlit as st  # noqa: PLC0415
        return st.secrets.get(key, default)
    except Exception:
        return default


@dataclass
class Settings:
    openrouter_api_key: str = field(
        default_factory=lambda: _get_secret("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = field(
        default_factory=lambda: _get_secret(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
    )
    default_model: str = field(
        default_factory=lambda: _get_secret(
            "DEFAULT_MODEL", "google/gemini-2.5-flash"
        )
    )
    supabase_api_url: str = field(
        default_factory=lambda: _get_secret("SUPABASE_API_URL", "")
    )
    supabase_key: str = field(
        default_factory=lambda: _get_secret("SUPABASE_SERVICE_ROLE_KEY", "")
    )
    supabase_storage_bucket: str = field(
        default_factory=lambda: _get_secret("SUPABASE_STORAGE_BUCKET", "documents")
    )
    sec_edgar_user_agent: str = field(
        default_factory=lambda: _get_secret("SEC_EDGAR_USER_AGENT", "market-comps myanagis@example.com")
    )
    sec_edgar_rate_limit_delay: float = field(
        default_factory=lambda: float(_get_secret("SEC_EDGAR_RATE_LIMIT_DELAY", "0.25"))
    )

    # OpenRouter model pricing (per 1M tokens) — used for cost estimation.
    # Keys match model IDs; values are (input_price_usd, output_price_usd).
    # These are approximate; check https://openrouter.ai/models for current rates.
    MODEL_PRICING: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "anthropic/claude-3-haiku": (0.2500, 1.2500),
            "anthropic/claude-3.5-haiku": (0.8000, 4.0000),
            "anthropic/claude-3.5-sonnet": (6.0000, 30.0000),
            "anthropic/claude-3.7-sonnet": (3.0000, 15.0000),
            "cohere/command-r-08-2024": (0.1500, 0.6000),
            "cohere/command-r-plus-08-2024": (2.5000, 10.0000),
            "deepseek/deepseek-chat": (0.3200, 0.8900),
            "deepseek/deepseek-r1": (0.7000, 2.5000),
            "google/gemini-2.0-flash-001": (0.1000, 0.4000),
            "google/gemini-2.0-flash-lite-001": (0.0750, 0.3000),
            "google/gemini-2.5-flash": (0.3000, 2.5000),
            "google/gemini-2.5-flash-lite": (0.1000, 0.4000),
            "google/gemini-2.5-pro": (1.2500, 10.0000),
            "meta-llama/llama-3.3-70b-instruct": (0.1000, 0.3200),
            "mistralai/mistral-large-2411": (2.0000, 6.0000),
            "mistralai/mixtral-8x7b-instruct": (0.5400, 0.5400),
            "openai/gpt-4o": (2.5000, 10.0000),
            "openai/gpt-4o-mini": (0.1500, 0.6000),
            "perplexity/llama-3.1-sonar-huge-128k-online": (5.0000, 5.0000),
        }
    )

    def get_model_pricing(self, model: str) -> tuple[float, float]:
        """Return (input_$/1M, output_$/1M) for the given model.
        Falls back to a safe conservative estimate if unknown."""
        return self.MODEL_PRICING.get(model, (1.00, 3.00))


# Singleton — import this anywhere
settings = Settings()

try:
    from supabase import create_client, Client
    supabase_client: Client | None = None
    if settings.supabase_api_url and settings.supabase_key:
        try:
            supabase_client = create_client(settings.supabase_api_url, settings.supabase_key)
        except Exception as e:
            print(f"Warning: Failed to initialize Supabase client: {e}")
            supabase_client = None
except ImportError:
    supabase_client = None

def get_supabase_url(file_path: str) -> str:
    """Returns a temporary signed URL for a file in Supabase storage."""
    if not supabase_client or not file_path:
        return ""
    try:
        res = supabase_client.storage.from_(settings.supabase_storage_bucket).create_signed_url(file_path, 3600)
        return res.get("signedURL") if isinstance(res, dict) else res
    except Exception:
        return ""

# Global default LLM Model
DEFAULT_LLM_MODEL: str = settings.default_model

# Centralised model list — import from here in all pages to maintain a single source of truth
MODEL_OPTIONS: list[str] = list(settings.MODEL_PRICING.keys())

DEFAULT_MODELS: list[str] = [
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",                
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat",            
]

DEFAULT_SUMMARY_MODEL: str = "openai/gpt-4o-mini"
