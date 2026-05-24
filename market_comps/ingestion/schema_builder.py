import os
import toml
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

SCHEMAS_DIR = os.path.join(os.path.dirname(__file__), '..', 'config', 'extraction_schemas')

def _type_mapping(toml_type: str) -> str:
    """Map custom TOML types to JSON Schema types."""
    mapping = {
        "currency": "number",
        "float": "number",
        "integer": "integer",
        "string": "string",
        "text": "string",
        "boolean": "boolean"
    }
    return mapping.get(toml_type, "string")

def load_toml_schema_as_json(schema_name: str) -> Optional[Dict[str, Any]]:
    """
    Load a TOML schema by name (e.g. 'company_revenue_metrics') 
    and convert it to a valid JSON schema for the LLM.
    Returns None if the schema file doesn't exist.
    """
    file_path = os.path.join(SCHEMAS_DIR, f"{schema_name}.toml")
    if not os.path.exists(file_path):
        return None
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
    except Exception as e:
        logger.error(f"Failed to parse TOML schema {file_path}: {e}")
        return None
        
    properties = {}
    required = []
    
    for field in config.get("fields", []):
        field_id = field.get("id")
        if not field_id:
            continue
            
        field_type = field.get("type", "string")
        
        if field_type == "array" and "items" in field:
            prop_def = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            if "description" in field:
                prop_def["description"] = field["description"]
                
            for item in field["items"]:
                i_id = item.get("id")
                if not i_id: continue
                i_type = _type_mapping(item.get("type", "string"))
                i_prop = {"type": i_type}
                if "description" in item:
                    i_prop["description"] = item["description"]
                prop_def["items"]["properties"][i_id] = i_prop
                if item.get("required"):
                    prop_def["items"]["required"].append(i_id)
            properties[field_id] = prop_def
        else:
            js_type = _type_mapping(field_type)
            prop_def = {"type": js_type}
            if "description" in field:
                prop_def["description"] = field["description"]
            properties[field_id] = prop_def
        
        if field.get("required", False):
            required.append(field_id)
            
    # Wrap in an array of items, because we might find multiple entities in a single document
    # For example, revenue metrics for 2024 and 2025
    json_schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        },
        "required": ["entities"]
    }
    
    return json_schema

def get_schema_description(schema_name: str) -> str:
    """Return the description of the schema to help build the prompt."""
    file_path = os.path.join(SCHEMAS_DIR, f"{schema_name}.toml")
    if not os.path.exists(file_path):
        return ""
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
            return config.get("description", "")
    except Exception:
        return ""

def build_aggregated_schema(schema_names: list[str]) -> tuple[Optional[Dict[str, Any]], str]:
    """
    Load multiple TOML schemas and merge them into a single comprehensive JSON Schema.
    Returns (json_schema, combined_description)
    """
    properties = {}
    required = set()
    descriptions = []
    
    for schema_name in schema_names:
        file_path = os.path.join(SCHEMAS_DIR, f"{schema_name}.toml")
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = toml.load(f)
        except Exception as e:
            logger.error(f"Failed to parse TOML schema {file_path}: {e}")
            continue
            
        desc = config.get("description")
        if desc:
            descriptions.append(desc)
            
        for field in config.get("fields", []):
            field_id = field.get("id")
            if not field_id:
                continue
                
            field_type = field.get("type", "string")
            
            if field_type == "array" and "items" in field:
                prop_def = {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
                if "description" in field:
                    prop_def["description"] = field["description"]
                    
                for item in field["items"]:
                    i_id = item.get("id")
                    if not i_id: continue
                    i_type = _type_mapping(item.get("type", "string"))
                    i_prop = {"type": i_type}
                    if "description" in item:
                        i_prop["description"] = item["description"]
                    prop_def["items"]["properties"][i_id] = i_prop
                    if item.get("required"):
                        prop_def["items"]["required"].append(i_id)
                properties[field_id] = prop_def
            else:
                js_type = _type_mapping(field_type)
                prop_def = {"type": js_type}
                if "description" in field:
                    prop_def["description"] = field["description"]
                properties[field_id] = prop_def
            
            if field.get("required", False):
                required.add(field_id)
                
    if not properties:
        return None, ""
        
    json_schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": list(required)
                }
            }
        },
        "required": ["entities"]
    }
    
    combined_desc = " ".join(descriptions)
    return json_schema, combined_desc
