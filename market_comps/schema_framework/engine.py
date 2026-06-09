import json
import logging
from pathlib import Path
from typing import Any, Dict

from market_comps.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Load static assets
_DIR = Path(__file__).resolve().parent
_PROMPT_PATH = _DIR / "evidence_extraction_prompt.md"

def _load_asset(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()

def parse_markdown_schema(md_text: str) -> dict:
    schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
    
    current_h1 = None
    current_h2 = None
    
    lines = md_text.split('\n')
    
    def create_evidence_array(description=""):
        return {
            "type": "array",
            "description": description.strip() if description.strip() else "Extracted evidence quotes",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string", "description": "Exact verbatim quote from the text."},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
                },
                "required": ["quote", "confidence"],
                "additionalProperties": False
            }
        }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('# '):
            header = line[2:].strip()
            current_h1 = header
            current_h2 = None
            schema["properties"][current_h1] = {
                "type": "object",
                "properties": {},
                "required": [],
                "description": "",
                "additionalProperties": False
            }
            if current_h1 not in schema["required"]:
                schema["required"].append(current_h1)
                
        elif line.startswith('## '):
            if current_h1 is None:
                continue
            header = line[3:].strip()
            current_h2 = header
            schema["properties"][current_h1]["properties"][current_h2] = create_evidence_array()
            schema["properties"][current_h1]["required"].append(current_h2)
            
        else:
            if current_h2 and current_h1:
                existing_desc = schema["properties"][current_h1]["properties"][current_h2].get("description", "")
                schema["properties"][current_h1]["properties"][current_h2]["description"] = (existing_desc + " " + line).strip()
            elif current_h1:
                existing_desc = schema["properties"][current_h1].get("description", "")
                schema["properties"][current_h1]["description"] = (existing_desc + " " + line).strip()
    
    for h1, h1_props in list(schema["properties"].items()):
        if not h1_props.get("properties"):
            desc = h1_props.get("description", "")
            schema["properties"][h1] = create_evidence_array(desc)
            
    return schema

def run_schema_extraction(
    text: str, 
    source_name: str, 
    source_date: str, 
    model: str,
    schema_text: str,
    extraction_prompt_template: str
) -> Dict[str, Any]:
    """
    Extracts structured evidence from raw text based on the schema framework.
    """
    prompt = extraction_prompt_template.replace("{{SCHEMA}}", schema_text).replace("{{TEXT}}", text)
    
    json_schema = parse_markdown_schema(schema_text)
    
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

def _build_synthesis_schema(extraction_schema: dict) -> dict:
    synth_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
    
    for k, v in extraction_schema["properties"].items():
        if v.get("type") == "object":
            # It has subheaders
            synth_schema["properties"][k] = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
            synth_schema["required"].append(k)
            for sub_k, sub_v in v["properties"].items():
                synth_schema["properties"][k]["properties"][sub_k] = {
                    "type": "string",
                    "description": f"Synthesized summary for {sub_k}"
                }
                synth_schema["properties"][k]["required"].append(sub_k)
        else:
            # It's an array directly
            synth_schema["properties"][k] = {
                "type": "string",
                "description": f"Synthesized summary for {k}"
            }
            synth_schema["required"].append(k)
            
    return synth_schema

def synthesize_evidence(combined_evidence: list[dict], model: str, synth_prompt_template: str, schema_text: str) -> tuple[dict, Any]:
    """
    Given a list of extracted evidence dictionaries, synthesize them into a coherent summary structured by schema.
    """
    evidence_for_synthesis = [
        {"source": e.get("source"), "date": e.get("date", ""), "data": e.get("data")} 
        for e in combined_evidence
    ]
    
    prompt = synth_prompt_template.replace("{{EVIDENCE_DATA}}", json.dumps(evidence_for_synthesis, indent=2))

    extraction_schema = parse_markdown_schema(schema_text)
    synth_schema = _build_synthesis_schema(extraction_schema)

    client = LLMClient(model=model)
    try:
        synthesis, usage = client.structured_output(
            prompt=prompt,
            json_schema=synth_schema,
            system_prompt="You are a data synthesis engine. Return strictly valid JSON.",
            model=model,
            temperature=0.3
        )
        return synthesis, usage
    except Exception as e:
        logger.error(f"Error synthesizing evidence: {e}")
        return {"error": f"Synthesis failed: {e}"}, None
