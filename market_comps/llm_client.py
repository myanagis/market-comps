# market_comps/llm_client.py
"""
Thin wrapper around the OpenAI-compatible OpenRouter API.
Provides structured JSON output and automatic token / cost tracking.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from openai import OpenAI
from pydantic import BaseModel

from market_comps.config import settings
from market_comps.models import LLMUsage

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Wraps OpenRouter via the openai SDK.

    Usage::

        client = LLMClient()
        result, usage = client.structured_output(
            prompt="List 5 cloud software companies...",
            json_schema={"type": "array", "items": {"type": "object", ...}},
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.model = model or settings.default_model

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or pass api_key= to LLMClient()."
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
        temperature: float = 0.2,
    ) -> tuple[str, LLMUsage]:
        """
        Call the chat completion endpoint.

        Returns:
            (content_string, LLMUsage) where LLMUsage tracks tokens + cost
            for THIS single call only (not cumulative).
        """
        model = model or self.model
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        if response_format:
            kwargs["response_format"] = response_format

        logger.debug("LLM call → model=%s, messages=%d", model, len(messages))
        response = self._client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""

        # Build usage record for this call
        usage = LLMUsage()
        if response.usage:
            in_price, out_price = settings.get_model_pricing(model)
            usage.add(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                input_price_per_m=in_price,
                output_price_per_m=out_price,
            )
        else:
            # OpenRouter may not always return usage; estimate conservatively
            logger.warning("No usage data returned for model %s", model)

        return content, usage

    def structured_output(
        self,
        prompt: str,
        json_schema: dict,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> tuple[Any, LLMUsage]:
        """
        Ask the model to return a JSON object matching json_schema.

        Returns:
            (parsed_python_object, LLMUsage)
        """
        system = system_prompt or (
            "You are a financial analyst assistant. "
            "Always respond with valid JSON that matches the requested schema. "
            "Do not include markdown code fences or any text outside the JSON."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        content, usage = self.chat_completion(
            messages=messages,
            model=model,
            response_format={"type": "json_object"},
            temperature=temperature,
        )

        # Strip any accidental markdown fences
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("```")[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON response: %s\nRaw: %s", exc, content[:500])
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

        return parsed, usage

    def simple_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> tuple[str, LLMUsage]:
        """Simple text completion — returns (text, LLMUsage)."""
        system = system_prompt or "You are a helpful financial analyst assistant."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.chat_completion(messages=messages, model=model, temperature=temperature)
