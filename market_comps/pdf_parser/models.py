# market_comps/pdf_parser/models.py
"""
Data models for the PDF Parser pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from market_comps.models import LLMUsage

MISTRAL_OCR_COST_PER_PAGE = 0.002  # $2 / 1,000 pages


# ---------------------------------------------------------------------------
# Per-term extraction result
# ---------------------------------------------------------------------------

@dataclass
class SupportingQuote:
    """A verbatim quote from the document with optional page reference."""
    text: str
    page: Optional[int] = None   # page number, if detectable


@dataclass
class ExtractedTerm:
    """One extracted field from a term sheet."""
    name: str
    value: Optional[str]
    """Extracted value, or None if not found / low confidence."""

    confidence: str
    """'high', 'low', or 'not_found'."""

    supporting_quotes: list[SupportingQuote] = field(default_factory=list)
    """Direct verbatim quotes from the document supporting the value."""

    possible_snippets: list[str] = field(default_factory=list)
    """Nearby snippets when value is None — may hint at the answer."""


# ---------------------------------------------------------------------------
# Overall parser result
# ---------------------------------------------------------------------------

@dataclass
class ParserResult:
    """Output from TermExtractor.run()."""

    filename: str
    pdf_engine: str
    model_used: str
    llm_usage: LLMUsage

    # Step 0 — classification
    document_type: str = "unknown"
    doc_type_confidence: str = "low"
    doc_type_rationale: str = ""

    # Step 1a — term extraction (term-sheet-like docs)
    terms: list[ExtractedTerm] = field(default_factory=list)

    # Step 1b — summary (non-term-sheet docs)
    summary: Optional[str] = None

    # PDF parsing metadata
    raw_extracted_text: Optional[str] = None
    pdf_pages: int = 0
    pdf_parsing_cost_usd: float = 0.0  # mistral-ocr: $0.002/page; pdf-text: free

    errors: list[str] = field(default_factory=list)
