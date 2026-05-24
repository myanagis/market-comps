import os

FILE_PATH = "pages/15_CRM_Document_Upload.py"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
in_processing_block = False

for i, line in enumerate(lines):
    if line.startswith("        if not text_content.strip():"):
        in_processing_block = True
    
    if in_processing_block and i > 105: # after the `if not text_content.strip():` block ends
        if line.startswith("            "):
            new_lines.append(line[8:])
        elif line == "\n":
            new_lines.append(line)
        elif line.startswith("# --- Display Results Outside of Submit Block ---"):
            in_processing_block = False
            new_lines.append(line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(FILE_PATH, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
