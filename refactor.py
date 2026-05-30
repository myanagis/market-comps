import os

REPLACEMENTS = {
    "IngestionRun": "PipelineRun",
    "ingestion_run": "pipeline_run",
    "ingestion_runs": "pipeline_runs",
    "CanonicalMutation": "AuditTrail",
    "canonical_mutation": "audit_trail",
    "canonical_mutations": "audit_trails",
    "pipeline_type": "connector_type"  # We'll just map old pipeline_type to connector_type for now so it doesn't break
}

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in REPLACEMENTS.items():
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

def main():
    dirs_to_search = ['market_comps', 'pages', 'scratch']
    for d in dirs_to_search:
        for root, dirs, files in os.walk(d):
            if '__pycache__' in root or 'alembic' in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    process_file(os.path.join(root, file))

if __name__ == '__main__':
    main()
