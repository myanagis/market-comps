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
            "DEFAULT_MODEL", "google/gemini-2.5-flash-lite-preview-06-17"
        )
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
            "x-ai/grok-4": (3.0000, 15.0000),
            "x-ai/grok-4-fast": (0.2000, 0.5000),
        }
    )

    def get_model_pricing(self, model: str) -> tuple[float, float]:
        """Return (input_$/1M, output_$/1M) for the given model.
        Falls back to a safe conservative estimate if unknown."""
        return self.MODEL_PRICING.get(model, (1.00, 3.00))


# Singleton — import this anywhere
settings = Settings()

# Centralised model list — import from here in all pages to maintain a single source of truth
MODEL_OPTIONS: list[str] = [
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3-haiku",
    "cohere/command-r-plus-08-2024",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large-2411",
    "mistralai/mixtral-8x7b-instruct",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "x-ai/grok-4",
    "x-ai/grok-4-fast",
]

DEFAULT_MODELS: list[str] = [
    "deepseek/deepseek-chat",            
    "x-ai/grok-4-fast",                  
    "openai/gpt-4o-mini",                
    "meta-llama/llama-3.3-70b-instruct", 
    "google/gemini-2.5-flash",
]

DEFAULT_SUMMARY_MODEL: str = "openai/gpt-4o-mini"
