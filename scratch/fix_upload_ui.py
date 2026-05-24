import os

FILE_PATH = "pages/15_CRM_Document_Upload.py"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
in_submitted = False
for i, line in enumerate(lines):
    if line.startswith("if submitted:"):
        new_lines.append("if submitted:\n")
        new_lines.append("    if not uploaded_file:\n")
        new_lines.append("        st.error(\"Please upload a file.\")\n")
        new_lines.append("    else:\n")
        new_lines.append("        st.session_state['start_processing'] = True\n")
        new_lines.append("        st.session_state['upload_file_name'] = uploaded_file.name\n")
        new_lines.append("        st.session_state['upload_file_bytes'] = uploaded_file.read()\n")
        new_lines.append("        st.session_state['upload_pdf_method'] = pdf_method\n")
        new_lines.append("        st.session_state['upload_custom_instructions'] = custom_instructions\n")
        new_lines.append("        st.session_state['upload_linked_org_id'] = linked_org_id\n")
        new_lines.append("\n")
        new_lines.append("if st.session_state.get('start_processing'):\n")
        new_lines.append("    st.session_state['start_processing'] = False  # Clear immediately to prevent infinite loops on unrelated reruns\n")
        new_lines.append("    file_name = st.session_state['upload_file_name']\n")
        new_lines.append("    file_bytes = st.session_state['upload_file_bytes']\n")
        new_lines.append("    pdf_method = st.session_state['upload_pdf_method']\n")
        new_lines.append("    custom_instructions = st.session_state['upload_custom_instructions']\n")
        new_lines.append("    linked_org_id = st.session_state['upload_linked_org_id']\n")
        new_lines.append("    with st.spinner(\"Processing document...\"):\n")
        
        in_submitted = True
        continue
        
    if in_submitted:
        if line.startswith("    if not uploaded_file:"):
            continue
        elif line.startswith("        st.error(\"Please upload a file.\")"):
            continue
        elif line.startswith("    else:"):
            continue
        elif line.startswith("        with st.spinner(\"Processing document...\"):"):
            continue
            
    if in_submitted and line.startswith("# --- Display Results Outside of Submit Block ---"):
        in_submitted = False
        
    if not in_submitted and not line.startswith("if submitted:"):
        # Adjust indentation if we are still inside the block, but my loop logic above bypasses the `else`
        if line.startswith("            ") and i < 255:
            # We un-indent by 4 spaces since we removed the `else:` block
            new_lines.append(line[4:])
        elif line.startswith("        ") and i < 255:
            new_lines.append(line[4:])
        else:
            new_lines.append(line)
    
with open(FILE_PATH, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
