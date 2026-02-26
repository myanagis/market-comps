# market_comps/pdf_parser/term_extractor.py
"""
Two-step agentic PDF pipeline:

  Step 0 — Classify  : Identify document type + extract paged text from annotations.
  Step 1a — Extract  : For term-sheet-like docs, pull structured fields + quotes w/ page refs.
  Step 1b — Summarize: For other docs, produce a plain-language summary.

Step 1 uses the paged text extracted from Step 0 annotations (no second PDF parse),
with page markers so the LLM can report which page each quote came from.
Falls back to re-sending pdf_bytes if no annotations were returned.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from market_comps.models import LLMUsage
from market_comps.pdf_parser.models import (
    ExtractedTerm, ParserResult, SupportingQuote, MISTRAL_OCR_COST_PER_PAGE
)
from market_comps.pdf_parser.pdf_client import PDFClient

logger = logging.getLogger(__name__)

# Document types that trigger structured term extraction
TERM_SHEET_TYPES = {"term_sheet", "safe_note", "convertible_note", "loi", "letter_of_intent"}

# Standard term sheet fields we want to extract
TERM_SHEET_FIELDS = [
    "Round",
    "Investment Type",
    "Pre-Money Valuation",
    "Post-Money Valuation",
    "Amount Raised / Investment Size",
    "Price Per Share",
    "Option Pool (Pre-Money)",
    "Option Pool (Post-Money)",
    "Liquidation Preference",
    "Participating Preferred",
    "Anti-Dilution Protection",
    "Dividend Rate",
    "Dividend Rate Calculation Type"
    "Board Composition",
    "Pro-Rata Rights",
    "Information Rights",
    "Drag-Along Rights",
    "Lead Investor",
    "Closing Date",
    "Company Counsel",
    "Investor Counsel",
    "Other Notable Provisions",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """\
You are a legal and financial document analyst.

Examine the document provided and identify what type of document it is.

Respond with ONLY valid JSON in exactly this format:
{
  "document_type": "<one of: term_sheet | safe_note | convertible_note | loi | other>",
  "confidence": "<high | medium | low>",
  "rationale": "<one sentence explaining your classification>"
}

Classification rules:
- term_sheet: A priced equity financing term sheet (Series A, Seed, etc.)
- safe_note: A Simple Agreement for Future Equity (SAFE)
- convertible_note: A convertible promissory note / bridge note
- loi: A letter of intent, MOU, or acquisition term sheet
- other: Any other document type (contracts, reports, etc.)
"""

_EXTRACT_PROMPT = """\
You are a senior venture capital analyst extracting key terms from a financing document.

STRICT RULES — read carefully before extracting:
- ONLY extract values that are EXPLICITLY AND LITERALLY written in the document below.
- Do NOT infer, estimate, calculate, or derive values from context.
- Do NOT assume standard market terms if they are not written in this document.
- Quotes MUST be copied verbatim, word-for-word from the document. No paraphrasing.
- When in doubt, use confidence "not_found". It is better to leave a field blank than to guess.
- If the document has page markers (=== Page N ===), include the page number in each quote.

DOCUMENT:
{document_text}

Fields to extract:
{fields_list}

Respond with ONLY valid JSON in exactly this format — no markdown, no prose:
{{
  "terms": [
    {{
      "name": "<field name exactly as listed above>",
      "value": "<value as written in document, or null>",
      "confidence": "<high | low | not_found>",
      "supporting_quotes": [
        {{"text": "<verbatim quote or quotes from document>", "page": <page number as integer, or null>}},
        ...
      ],
      "possible_snippets": ["<nearby text when uncertain>", ...]
    }},
    ...
  ]
}}
"""

_SUMMARIZE_PROMPT = """\
You are a senior analyst. Summarize the key points of this document in clear, concise prose.
Only include information that is explicitly present in the document — do not speculate.

DOCUMENT:
{document_text}

Include:
- What type of document this is
- The parties involved (if identifiable)
- The main subject matter or purpose
- Any key dates, amounts, or obligations mentioned
- Any notable terms or conditions

Write 3-6 paragraphs. Be factual and neutral.
"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class TermExtractor:
    """
    Orchestrates the two-step PDF parsing pipeline.

    Step 0 sends the PDF, gets file_annotations back, and extracts paged text.
    Step 1 sends the paged text as plain context (no second PDF parse).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        pdf_engine: str = "pdf-text",
    ) -> None:
        self._client = PDFClient(api_key=api_key, model=model, pdf_engine=pdf_engine)
        self.model = self._client.model
        self.pdf_engine = pdf_engine

    def run(self, pdf_bytes: bytes, filename: str = "document.pdf") -> ParserResult:
        """Run the full classify → extract/summarize pipeline."""
        cumulative_usage = LLMUsage()
        result = ParserResult(
            filename=filename,
            pdf_engine=self.pdf_engine,
            model_used=self.model,
            llm_usage=cumulative_usage,
        )

        # ── Step 0: Classify + extract paged text ────────────────────────────
        logger.info("TermExtractor: Step 0 — classifying %s", filename)
        file_annotations = None
        paged_text: Optional[str] = None
        try:
            classify_content, file_annotations, classify_usage = self._client.send(
                prompt=_CLASSIFY_PROMPT,
                pdf_bytes=pdf_bytes,
                filename=filename,
                temperature=0.0,
            )
            _merge_usage(cumulative_usage, classify_usage)

            classification = _parse_json(classify_content)
            result.document_type = classification.get("document_type", "other")
            result.doc_type_confidence = classification.get("confidence", "low")
            result.doc_type_rationale = classification.get("rationale", "")
            logger.info("Classified as: %s (%s)", result.document_type, result.doc_type_confidence)

            # Extract paged text + page count from annotations
            paged_text, page_count = _extract_paged_text(file_annotations)

            # fallback: count pages from PDF bytes (annotations may return one big block)
            if page_count <= 1 and pdf_bytes:
                counted = _count_pdf_pages(pdf_bytes)
                if counted > page_count:
                    page_count = counted

            result.raw_extracted_text = paged_text
            result.pdf_pages = page_count

            # Calculate PDF parsing cost (mistral-ocr only)
            if self.pdf_engine == "mistral-ocr" and page_count > 0:
                result.pdf_parsing_cost_usd = page_count * MISTRAL_OCR_COST_PER_PAGE
            else:
                result.pdf_parsing_cost_usd = 0.0

        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            result.errors.append(f"Classification error: {exc}")
            return result

        # Choose document text for Step 1: prefer paged_text; fall back to re-sending PDF
        doc_text = paged_text or "[Document text unavailable — see raw PDF]"
        use_fallback_pdf = paged_text is None

        # ── Step 1a: Extract terms (term-sheet-like) ─────────────────────────
        if result.document_type in TERM_SHEET_TYPES:
            logger.info("TermExtractor: Step 1a — extracting terms (use_fallback_pdf=%s)", use_fallback_pdf)
            try:
                fields_list = "\n".join(f"- {f}" for f in TERM_SHEET_FIELDS)
                extract_prompt = _EXTRACT_PROMPT.format(
                    document_text=doc_text,
                    fields_list=fields_list,
                )

                if use_fallback_pdf:
                    extract_content, _, extract_usage = self._client.send(
                        prompt=extract_prompt,
                        pdf_bytes=pdf_bytes,
                        filename=filename,
                        temperature=0.0,
                    )
                else:
                    # Text-only call — no PDF re-parse, cheaper
                    extract_content, _, extract_usage = self._client.send(
                        prompt=extract_prompt,
                        temperature=0.0,
                    )
                _merge_usage(cumulative_usage, extract_usage)

                extracted = _parse_json(extract_content)
                result.terms = _parse_terms(extracted.get("terms", []))

            except Exception as exc:
                logger.error("Extraction failed: %s", exc)
                result.errors.append(f"Extraction error: {exc}")

        # ── Step 1b: Summarize (all other document types) ────────────────────
        else:
            logger.info("TermExtractor: Step 1b — summarizing")
            try:
                summarize_prompt = _SUMMARIZE_PROMPT.format(document_text=doc_text)

                if use_fallback_pdf:
                    summary_content, _, summary_usage = self._client.send(
                        prompt=summarize_prompt,
                        pdf_bytes=pdf_bytes,
                        filename=filename,
                        temperature=0.3,
                    )
                else:
                    summary_content, _, summary_usage = self._client.send(
                        prompt=summarize_prompt,
                        temperature=0.3,
                    )
                _merge_usage(cumulative_usage, summary_usage)
                result.summary = summary_content.strip()

            except Exception as exc:
                logger.error("Summarization failed: %s", exc)
                result.errors.append(f"Summarization error: {exc}")

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_usage(target: LLMUsage, source: LLMUsage) -> None:
    target.total_prompt_tokens += source.total_prompt_tokens
    target.total_completion_tokens += source.total_completion_tokens
    target.total_tokens += source.total_tokens
    target.estimated_cost_usd += source.estimated_cost_usd
    target.call_count += source.call_count


def _parse_json(content: str) -> dict:
    """Strip markdown fences and parse JSON."""
    s = content.strip()
    if s.startswith("```"):
        s = s.split("```", 1)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0].strip()
    return json.loads(s)


def _extract_paged_text(annotations: Optional[list[dict]]) -> tuple[Optional[str], int]:
    """
    Build page-numbered text from OpenRouter file_annotations.
    Returns (paged_text, page_count).

    For pdf-text: one large text block → treated as page 1.
    For mistral-ocr: one content part per page → numbered accordingly.
    Falls back to raw annotation JSON if no text found.
    """
    if not annotations:
        return None, 0

    parts: list[str] = []
    page_num = 1

    for annotation in annotations:
        if annotation.get("type") != "file":
            continue
        for content_part in annotation.get("file", {}).get("content", []):
            if content_part.get("type") != "text":
                continue
            text = content_part.get("text", "").strip()
            # Strip <file name="...">...</file> XML wrapper if present
            text = re.sub(r"^<file[^>]*>", "", text).strip()
            text = re.sub(r"</file>\s*$", "", text).strip()
            if text:
                parts.append(f"=== Page {page_num} ===\n{text}")
                page_num += 1

    if parts:
        return "\n\n".join(parts), len(parts)

    # Fallback: show raw annotation JSON for debugging
    import json as _json
    return _json.dumps(annotations, indent=2), 0


def _parse_terms(raw: list[dict]) -> list[ExtractedTerm]:
    terms = []
    for item in raw:
        raw_quotes = item.get("supporting_quotes", [])
        parsed_quotes: list[SupportingQuote] = []
        for q in raw_quotes:
            if isinstance(q, dict):
                parsed_quotes.append(SupportingQuote(
                    text=q.get("text", ""),
                    page=q.get("page"),  # int or None
                ))
            elif isinstance(q, str):
                # Backwards compat: plain string quote
                parsed_quotes.append(SupportingQuote(text=q))

        terms.append(ExtractedTerm(
            name=item.get("name", "Unknown"),
            value=item.get("value"),
            confidence=item.get("confidence", "not_found"),
            supporting_quotes=parsed_quotes,
            possible_snippets=item.get("possible_snippets", []),
        ))
    return terms
def _count_pdf_pages(pdf_bytes: bytes) -> int:
    """Count pages in a PDF from binary content."""
    try:
        # Try the most common patterns — whitespace between /Type and /Page varies
        counts = [
            len(re.findall(rb"/Type\s*/Page[^s]", pdf_bytes)),
            len(re.findall(rb"/Type /Page[\s/]", pdf_bytes)),
            len(re.findall(rb"/Type/Page[\s/]", pdf_bytes)),
        ]
        return max(counts)
    except Exception:
        return 0
