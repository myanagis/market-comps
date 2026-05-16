import json
import logging
from typing import Dict, Any

from prefect import task
from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage

logger = logging.getLogger(__name__)

_CLASSIFY_PROMPT = """\
You are a legal and financial document analyst.

Examine the document provided and identify what type of document it is.

Respond with ONLY valid JSON in exactly this format:
{{
  "document_type": "<one of: term_sheet | safe_note | convertible_note | loi | presentation | other>",
  "confidence": "<high | medium | low>",
  "rationale": "<one sentence explaining your classification>"
}}

Classification rules:
- term_sheet: A priced equity financing term sheet (Series A, Seed, etc.)
- safe_note: A Simple Agreement for Future Equity (SAFE)
- convertible_note: A convertible promissory note / bridge note
- loi: A letter of intent, MOU, or acquisition term sheet
- presentation: A pitch deck, presentation, or slides
- other: Any other document type (contracts, reports, etc.)

DOCUMENT:
{document_text}
"""

@task(name="classify_document")
def classify_document(document_text: str, model: str) -> tuple[Dict[str, Any], LLMUsage]:
    """
    Classify a document into a predefined type using an LLM.
    Returns the JSON classification and LLMUsage.
    """
    client = LLMClient(model=model)
    prompt = _CLASSIFY_PROMPT.format(document_text=document_text)
    
    # We use chat_completion here with JSON formatting request if needed, 
    # but simple_text + json.loads is fine if we force JSON structure.
    response_text, usage = client.simple_text(prompt, temperature=0.0)
    
    # Clean response text
    clean_text = response_text.replace("```json", "").replace("```", "").strip()
    try:
        classification = json.loads(clean_text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse classification JSON. Defaulting to 'other'.")
        classification = {
            "document_type": "other",
            "confidence": "low",
            "rationale": "Failed to parse JSON response."
        }
        
    return classification, usage
