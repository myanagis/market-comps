import json
import logging
from pathlib import Path
from typing import Any, Dict

from market_comps.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Load static assets
_DIR = Path(__file__).resolve().parent
_SCHEMA_PATH = _DIR / "starter_schema.json"
_PROMPT_PATH = _DIR / "evidence_extraction_prompt.md"

def _load_asset(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()

def run_schema_extraction(text: str, source_name: str, source_date: str, model: str) -> Dict[str, Any]:
    """
    Extracts structured evidence from raw text based on the schema framework.
    """
    schema_text = _load_asset(_SCHEMA_PATH)
    prompt_template = _load_asset(_PROMPT_PATH)

    # Substitute variables
    prompt = prompt_template.replace("{{SCHEMA}}", schema_text).replace("{{TEXT}}", text)
    
    # We will use the schema provided as the expected output format for structured_output
    json_schema = json.loads(schema_text)
    
    client = LLMClient(model=model)
    
    try:
        parsed_data, usage = client.structured_output(
            prompt=prompt,
            json_schema=json_schema,
            system_prompt="You are an evidence extraction engine. Return strictly valid JSON.",
            model=model,
            temperature=0.1
        )
    except Exception as e:
        logger.error(f"Error extracting evidence for source '{source_name}': {e}")
        parsed_data = {"error": str(e)}
        usage = None

    return {
        "source": source_name,
        "date": source_date,
        "company": "Extracted",
        "data": parsed_data,
        "usage": usage
    }

def synthesize_evidence(combined_evidence: list[dict], model: str) -> str:
    """
    Given a list of extracted evidence dictionaries, synthesize them into a coherent summary.
    """
    # Filter out non-serializable or irrelevant fields like LLMUsage
    evidence_for_synthesis = [
        {"source": e.get("source"), "date": e.get("date", ""), "data": e.get("data")} 
        for e in combined_evidence
    ]
    
    synth_prompt_path = _DIR / "synthesis_prompt.md"
    prompt_template = _load_asset(synth_prompt_path)
    prompt = prompt_template.replace("{{EVIDENCE_DATA}}", json.dumps(evidence_for_synthesis, indent=2))

    client = LLMClient(model=model)
    try:
        synthesis, usage = client.simple_text(
            prompt=prompt,
            model=model,
            temperature=0.3
        )
        return synthesis, usage
    except Exception as e:
        logger.error(f"Error synthesizing evidence: {e}")
        return f"Synthesis failed: {e}", None
