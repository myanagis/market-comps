import streamlit as st
import os
import re

st.title("Directory Analyzer")
st.write("View the contents of any directory on your system.")

# Get directory input from user
target_dir = st.text_input("Enter an absolute directory path:", value="", placeholder="e.g., C:\\Users\\micha\\Documents")

# Natural sort logic: extracts numbers from string so "10" is sorted after "2"
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

if target_dir:
    if not os.path.exists(target_dir):
        st.error(f"The path '{target_dir}' does not exist.")
    elif not os.path.isdir(target_dir):
        st.error(f"The path '{target_dir}' is not a directory.")
    else:
        st.write(f"### Contents of `{target_dir}`")
        
        def build_tree_html(directory):
            html = ""
            
            try:
                # Separate items into directories and files
                items = os.listdir(directory)
                folders = []
                files = []
                for item in items:
                    item_path = os.path.join(directory, item)
                    if os.path.isdir(item_path):
                        folders.append(item)
                    else:
                        files.append(item)
                        
                # Use natural sort to order "1", "2", "10" correctly
                folders.sort(key=natural_sort_key)
                files.sort(key=natural_sort_key)
                
                # Render folders using HTML <details> tag for collapsibility
                for folder in folders:
                    folder_path = os.path.join(directory, folder)
                    html += f'<details open style="margin-bottom: 4px;"><summary style="cursor: pointer; font-weight: bold; color: #666666;"><span style="color: #808080; margin-right: 8px;">📁</span>{folder}</summary><div style="margin-left: 24px; margin-top: 4px;">\n'
                    # Recursive call
                    html += build_tree_html(folder_path)
                    html += '</div></details>\n'
                    
                # Render files
                for file in files:
                    html += f'<div style="margin-bottom: 4px; display: flex; align-items: center;"><span style="color: #999999; margin-right: 8px;">📄</span><span style="color: #111111;">{file}</span></div>\n'
                    
            except PermissionError:
                html += f'<div style="color: #cc0000; font-size: 0.9em; margin-bottom: 4px;">🚫 Permission Denied</div>\n'
            except Exception as e:
                html += f'<div style="color: #cc0000; font-size: 0.9em; margin-bottom: 4px;">⚠️ Error: {e}</div>\n'
                
            return html

        with st.spinner("Generating tree view..."):
            tree_html = build_tree_html(target_dir)
            
            # Wrap the tree in a styled container so black text is always visible
            if tree_html:
                container_html = f'<div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 14px; line-height: 1.5; max-height: 600px; overflow-y: auto; overflow-x: auto;">\n{tree_html}\n</div>'
                st.markdown(container_html, unsafe_allow_html=True)
            else:
                st.info("Directory is empty.")
