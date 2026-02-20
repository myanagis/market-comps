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
            "DEFAULT_MODEL", "google/gemini-flash-1.5"
        )
    )

    # OpenRouter model pricing (per 1M tokens) — used for cost estimation.
    # Keys match model IDs; values are (input_price_usd, output_price_usd).
    # These are approximate; check https://openrouter.ai/models for current rates.
    MODEL_PRICING: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "google/gemini-flash-1.5": (0.075, 0.30),
            "google/gemini-flash-1.5-8b": (0.0375, 0.15),
            "google/gemini-2.0-flash-001": (0.10, 0.40),
            "openai/gpt-4o-mini": (0.15, 0.60),
            "openai/gpt-4o": (2.50, 10.00),
            "anthropic/claude-3-haiku": (0.25, 1.25),
            "anthropic/claude-3.5-sonnet": (3.00, 15.00),
            "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),
        }
    )

    def get_model_pricing(self, model: str) -> tuple[float, float]:
        """Return (input_$/1M, output_$/1M) for the given model.
        Falls back to a safe conservative estimate if unknown."""
        return self.MODEL_PRICING.get(model, (1.00, 3.00))


# Singleton — import this anywhere
settings = Settings()
