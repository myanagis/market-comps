import os

FILE_PATH = "pages/15_CRM_Document_Upload.py"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # The replaced block ends around line 102.
    # The rest of the `with st.spinner:` block goes up to line 265.
    if i > 102 and i < 265:
        # Check if line starts with 8 spaces
        if line.startswith("        "):
            new_lines.append(line[8:])
        elif line == "\n" or line.strip() == "":
            new_lines.append(line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(FILE_PATH, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Unindented lines 103 to 264 by 8 spaces.")
