import os
import toml

CONFIG_DIR = os.path.join("market_comps", "config")
CLASSIFIER_FILE = os.path.join(CONFIG_DIR, "document_classifier.toml")
SCHEMAS_DIR = os.path.join(CONFIG_DIR, "extraction_schemas")

with open(CLASSIFIER_FILE, "r", encoding="utf-8") as f:
    classifier_config = toml.load(f)

for doc_class in classifier_config.get("classes", []):
    class_id = doc_class.get("id")
    recommended = doc_class.get("recommended_schemas", [])
    optional = doc_class.get("optional_schemas", [])
    
    all_schemas = recommended + optional
    if not all_schemas:
        continue
        
    merged_fields = []
    seen_field_ids = set()
    
    for schema_name in all_schemas:
        schema_file = os.path.join(SCHEMAS_DIR, f"{schema_name}.toml")
        if not os.path.exists(schema_file):
            print(f"WARNING: Schema file {schema_file} does not exist.")
            continue
            
        with open(schema_file, "r", encoding="utf-8") as sf:
            s_config = toml.load(sf)
            
        for field in s_config.get("fields", []):
            field_id = field.get("id")
            if field_id not in seen_field_ids:
                merged_fields.append(field)
                seen_field_ids.add(field_id)
                
    # Create new consolidated schema
    new_schema = {
        "name": class_id,
        "version": "v1",
        "description": doc_class.get("description", f"Extracted fields for {class_id}"),
        "enabled": True,
        "model_complexity": doc_class.get("extraction_complexity", "medium"),
        "fields": merged_fields
    }
    
    new_schema_path = os.path.join(SCHEMAS_DIR, f"{class_id}.toml")
    with open(new_schema_path, "w", encoding="utf-8") as out_f:
        toml.dump(new_schema, out_f)
        
    print(f"Created {new_schema_path} with {len(merged_fields)} fields.")
