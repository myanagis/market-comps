# market_comps/pdf_parser/pdf_client.py
"""
Low-level OpenRouter client for PDF inputs.

Sends a base64-encoded PDF alongside a prompt and returns the raw text response,
any file_annotations from the response, and an LLMUsage record.

On subsequent calls, if file_annotations from the first response and the original
pdf_bytes are both supplied, OpenRouter can skip re-parsing the PDF.

Supports three PDF engines:
  - "pdf-text"   : Free, plain text extraction (best for machine-generated PDFs)
  - "mistral-ocr": $2 / 1,000 pages — best for scanned / image-heavy PDFs
  - "native"     : Model-native file handling (charged as input tokens)
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Optional

import requests

from market_comps.config import settings
from market_comps.models import LLMUsage

logger = logging.getLogger(__name__)

PDF_ENGINE_PRICING: dict[str, str] = {
    "pdf-text": "Free",
    "mistral-ocr": "$2 / 1,000 pages",
    "native": "Input tokens",
}


class PDFClient:
    """
    Thin wrapper that sends PDF content to OpenRouter via the file-parser plugin.

    Parameters
    ----------
    api_key : str, optional
        OpenRouter API key. Defaults to settings.openrouter_api_key.
    model : str, optional
        Model to use for completion. Defaults to settings.default_model.
    pdf_engine : str
        One of "pdf-text" (default), "mistral-ocr", "native".
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        pdf_engine: str = "pdf-text",
    ) -> None:
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.default_model
        self.pdf_engine = pdf_engine

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file."
            )

        self._base_url = settings.openrouter_base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "Market Comps PDF Parser",
        }
        # Cached data_url from first call — used to reconstruct annotation messages
        self._cached_data_url: Optional[str] = None
        self._cached_filename: str = "document.pdf"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        prompt: str,
        pdf_bytes: Optional[bytes] = None,
        filename: str = "document.pdf",
        file_annotations: Optional[list[dict]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> tuple[str, Optional[list[dict]], LLMUsage]:
        """
        Send a prompt to the LLM with PDF content.

        First call: provide `pdf_bytes`. The client caches the base64 data_url
        internally for annotation reuse.

        Subsequent calls: provide `file_annotations` from the first response.
        The client reconstructs the correct message structure (original file msg
        + assistant annotations + new prompt) so OpenRouter skips re-parsing.

        Note: response_format is intentionally NOT used here — not all models
        support json_object mode. JSON output is enforced via prompt instruction.

        Returns
        -------
        (content, file_annotations, LLMUsage)
        """
        if pdf_bytes is not None:
            b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            self._cached_data_url = f"data:application/pdf;base64,{b64}"
            self._cached_filename = filename

        messages = self._build_messages(
            prompt=prompt,
            filename=filename,
            file_annotations=file_annotations,
            system_prompt=system_prompt,
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        # Add file-parser plugin on the first call (when we're actually uploading)
        if pdf_bytes is not None and self.pdf_engine != "native":
            payload["plugins"] = [
                {"id": "file-parser", "pdf": {"engine": self.pdf_engine}}
            ]

        logger.debug(
            "PDFClient: POST %s/chat/completions model=%s engine=%s annotations=%s",
            self._base_url, self.model, self.pdf_engine,
            "yes" if file_annotations else "no",
        )

        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=120,
        )

        if not resp.ok:
            # Log response body so we can debug 4xx errors
            logger.error("OpenRouter %d: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()

        data = resp.json()
        choice = data["choices"][0]
        content: str = choice["message"].get("content") or ""
        new_annotations: Optional[list[dict]] = choice["message"].get("annotations")

        # Build usage
        usage = LLMUsage()
        raw_usage = data.get("usage", {})
        if raw_usage:
            in_price, out_price = settings.get_model_pricing(self.model)
            usage.add(
                prompt_tokens=raw_usage.get("prompt_tokens", 0),
                completion_tokens=raw_usage.get("completion_tokens", 0),
                input_price_per_m=in_price,
                output_price_per_m=out_price,
            )

        return content, new_annotations, usage

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        filename: str,
        file_annotations: Optional[list[dict]],
        system_prompt: Optional[str],
    ) -> list[dict]:
        messages: list[dict] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if file_annotations and self._cached_data_url:
            # Annotation-reuse pattern (per OpenRouter docs):
            # 1. Original user message WITH the file
            # 2. Assistant message WITH the annotations
            # 3. New user message with the actual prompt
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyse this document."},
                    {
                        "type": "file",
                        "file": {
                            "filename": self._cached_filename,
                            "file_data": self._cached_data_url,
                        },
                    },
                ],
            })
            messages.append({
                "role": "assistant",
                "content": "I have read and parsed the document.",
                "annotations": file_annotations,
            })
            messages.append({"role": "user", "content": prompt})
        else:
            # First call — include the file directly
            user_content: list[dict] = [{"type": "text", "text": prompt}]
            if self._cached_data_url:
                user_content.append({
                    "type": "file",
                    "file": {
                        "filename": self._cached_filename,
                        "file_data": self._cached_data_url,
                    },
                })
            messages.append({"role": "user", "content": user_content})

        return messages
