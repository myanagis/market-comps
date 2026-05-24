import os
import toml
import logging
from typing import Tuple, Dict, Any

from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'document_classifier.toml')

def load_classifier_config() -> dict:
    """Load the TOML configuration for document classification."""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return toml.load(f)
    except Exception as e:
        logger.error(f"Failed to load document_classifier.toml: {e}")
        return {}

def classify_document(text_content: str, llm_client: LLMClient) -> Tuple[Dict[str, Any], LLMUsage]:
    """
    Classify a document using the LLM based on classes defined in TOML.
    
    Returns:
        Tuple of (classification_result_dict, LLMUsage)
        e.g. ({"document_type": "startup_pitch_deck", "confidence": 0.95, ...}, usage)
    """
    config = load_classifier_config()
    if not config:
        raise ValueError("Classifier config is empty or failed to load.")
        
    classes = config.get("classes", [])
    if not classes:
        raise ValueError("No document classes defined in TOML.")
        
    # Build prompt context
    classes_context = []
    for cls in classes:
        if not cls.get("enabled", True):
            continue
        classes_context.append(
            f"Class ID: {cls['id']}\n"
            f"Description: {cls.get('description', '')}\n"
            f"Examples: {', '.join(cls.get('examples', []))}\n"
        )
        
    class_ids = [c["id"] for c in classes if c.get("enabled", True)]
    if config.get("allow_unknown", True) and "unknown" not in class_ids:
        class_ids.append("unknown")
        classes_context.append("Class ID: unknown\nDescription: Document could not confidently be classified.")
        
    system_prompt = (
        "You are an expert document classifier. "
        "Your job is to read the provided text and classify it into exactly one of the provided Class IDs.\n\n"
        "Here are the available document classes:\n"
        "----------------------------------------\n"
        f"{chr(10).join(classes_context)}\n"
        "----------------------------------------\n"
        "Analyze the content carefully. Output ONLY valid JSON matching EXACTLY this schema:\n"
        "{\n"
        '  "document_type": "string (the class id)",\n'
        '  "confidence": 0.95,\n'
        '  "candidate_classes": ["other_class", "another_class"],\n'
        '  "reasoning": "string"\n'
        "}\n"
        "Do not include any other keys. 'document_type' is required and must be the exact Class ID."
    )
    
    # Truncate text if it's too long to save cost/tokens on classification
    # 20,000 characters is usually enough to classify a document
    truncated_text = text_content[:20000]
    
    user_prompt = f"Please classify the following document excerpt:\n\n{truncated_text}"
    
    schema = {
        "type": "object",
        "properties": {
            "document_type": {
                "type": "string", 
                "enum": class_ids,
                "description": "The selected class ID that best fits the document."
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0.0 and 1.0"
            },
            "candidate_classes": {
                "type": "array",
                "items": {"type": "string", "enum": class_ids},
                "description": "Other potential classes considered"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this class was chosen."
            }
        },
        "required": ["document_type", "confidence", "candidate_classes"]
    }
    
    result, usage = llm_client.structured_output(
        prompt=user_prompt,
        json_schema=schema,
        system_prompt=system_prompt,
        temperature=0.1,
        step_name="document_classification"
    )
    
    return result, usage

def get_recommended_schemas(document_type: str) -> list[str]:
    """Retrieve recommended schemas for a specific document type."""
    config = load_classifier_config()
    classes = config.get("classes", [])
    
    for cls in classes:
        if cls["id"] == document_type:
            return cls.get("recommended_schemas", [])
            
    return []
