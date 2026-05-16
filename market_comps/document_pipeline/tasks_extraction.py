import json
import logging
from typing import Dict, Any, Tuple

from prefect import task
from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage
from market_comps.pdf_parser.models import ParserResult, ExtractedTerm, SupportingQuote

logger = logging.getLogger(__name__)

# Document types that trigger structured term extraction
TERM_SHEET_TYPES = {"term_sheet", "safe_note", "convertible_note", "loi"}

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
    "Dividend Rate Calculation Type",
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
        {{"text": "<verbatim quote or quotes from document>", "page": <page number as integer, or null>}}
      ],
      "possible_snippets": ["<nearby text when uncertain>"]
    }}
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

@task(name="extract_data")
def extract_data(document_text: str, classification: Dict[str, Any], model: str) -> Tuple[ParserResult, LLMUsage]:
    """
    Extract structured data or a summary depending on the document classification.
    """
    doc_type = classification.get("document_type", "other")
    client = LLMClient(model=model)
    
    result = ParserResult(
        document_type=doc_type,
        doc_type_confidence=classification.get("confidence", "low"),
        doc_type_rationale=classification.get("rationale", ""),
        raw_extracted_text=document_text,
        summary="",
        terms=[],
        errors=[],
        model_used=model,
        pdf_engine="n/a", # Will be set by flow
        llm_usage=LLMUsage()
    )

    if doc_type in TERM_SHEET_TYPES:
        prompt = _EXTRACT_PROMPT.format(
            document_text=document_text,
            fields_list=", ".join(TERM_SHEET_FIELDS)
        )
        response_text, usage = client.simple_text(prompt, temperature=0.0)
        
        # Clean response
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(clean_text)
            for t in parsed.get("terms", []):
                quotes = []
                for sq in t.get("supporting_quotes", []):
                    # handle old dict format or newer object format
                    if isinstance(sq, dict):
                        quotes.append(SupportingQuote(
                            text=sq.get("text", ""),
                            page=sq.get("page")
                        ))
                    elif isinstance(sq, str):
                        quotes.append(SupportingQuote(text=sq, page=None))
                        
                result.terms.append(
                    ExtractedTerm(
                        name=t.get("name", ""),
                        value=t.get("value"),
                        confidence=t.get("confidence", "low"),
                        supporting_quotes=quotes,
                        possible_snippets=t.get("possible_snippets", [])
                    )
                )
        except json.JSONDecodeError as e:
            logger.error("Failed to parse extraction JSON.")
            result.errors.append(f"Failed to parse term extraction: {e}")
            
    else:
        # Standard summary
        prompt = _SUMMARIZE_PROMPT.format(document_text=document_text)
        summary, usage = client.simple_text(prompt, temperature=0.0)
        result.summary = summary

    return result, usage
