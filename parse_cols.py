import json

file_path = r'C:\Users\micha\.gemini\antigravity-ide\brain\9e2e52d3-1825-48b9-93f6-43dd0f1489cf\.system_generated\steps\67\content.md'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

json_str = content.split('---\n\n')[1]
try:
    cols = json.loads(json_str)
    for c in cols:
        print(f"{c.get('fieldName', '')}: {c.get('name', '')} - {c.get('description', '')}")
except Exception as e:
    print(f"Error parsing JSON: {e}")
