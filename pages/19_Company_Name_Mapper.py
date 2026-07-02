import streamlit as st
import pandas as pd
import re
from io import BytesIO

try:
    from rapidfuzz import fuzz
except ImportError:
    st.error("Please add 'rapidfuzz' to your requirements.txt and install it.")
    st.stop()

st.set_page_config(page_title="Company Name Mapper", page_icon="🏷️", layout="wide")
st.title("🏷️ Company Name Mapper")
st.markdown("Paste a list of company names or upload an Excel file to automatically map them to canonical names.")

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.strip()
    if not n: return ""
    
    # Work in uppercase for reliable regex boundaries
    n = n.upper()
    
    # Treat DBA, d/b/a, FKA, AKA, formerly as alias indicators
    alias_keywords = [r'\bDBA\b', r'\bD/B/A\b', r'\bFKA\b', r'\bAKA\b', r'\bFORMERLY\b']
    for kw in alias_keywords:
        parts = re.split(kw, n, flags=re.IGNORECASE)
        if len(parts) > 1:
            n = parts[0] # Take the prefix before the alias indicator
            break
            
    # Remove punctuation
    n = re.sub(r'[^\w\s]', '', n)
    
    # Remove weak suffixes/terms
    suffixes = [
        r'\bLLC\b', r'\bINC\b', r'\bINCORPORATED\b', r'\bCORP\b', 
        r'\bCORPORATION\b', r'\bLTD\b', r'\bLIMITED\b', r'\bCO\b', 
        r'\bCOMPANY\b', r'\bHOLDINGS\b', r'\bGROUP\b'
    ]
    for suf in suffixes:
        n = re.sub(suf, '', n, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    n = re.sub(r'\s+', ' ', n)
    
    # Upper first letter (Title case)
    return n.strip().title()

def map_companies(names: list[str]) -> pd.DataFrame:
    canonical_list = [] # [{"canonical": "Meta", "normalized": "Meta"}]
    results = []
    
    for orig in names:
        orig_str = str(orig).strip()
        if not orig_str:
            continue
            
        norm = normalize_name(orig_str)
        
        if not canonical_list:
            canonical_list.append({"canonical": orig_str, "normalized": norm})
            results.append({
                "original_name": orig_str,
                "canonical_name": orig_str,
                "score": 100.0,
                "status": "new"
            })
            continue
            
        best_match = None
        best_score = -1
        
        for can in canonical_list:
            score1 = fuzz.token_set_ratio(norm, can["normalized"])
            score2 = fuzz.partial_ratio(norm, can["normalized"])
            score = max(score1, score2)
            
            if score > best_score:
                best_score = score
                best_match = can
                
        if best_score >= 95:
            results.append({
                "original_name": orig_str,
                "canonical_name": best_match["canonical"],
                "score": round(best_score, 1),
                "status": "auto_mapped"
            })
        elif best_score >= 75:
            results.append({
                "original_name": orig_str,
                "canonical_name": best_match["canonical"],
                "score": round(best_score, 1),
                "status": "review"
            })
        else:
            # Create new canonical
            canonical_list.append({"canonical": orig_str, "normalized": norm})
            results.append({
                "original_name": orig_str,
                "canonical_name": orig_str,
                "score": round(best_score, 1) if best_match else 0.0,
                "status": "new"
            })
            
    return pd.DataFrame(results)

# UI
st.sidebar.header("Configuration")
input_method = st.sidebar.radio("Input Method", ["Paste Text", "Upload Excel"])

names = []
if input_method == "Paste Text":
    text = st.text_area("Paste company names (one per line)", height=250, placeholder="Meta\nMeta DBA Facebook\nMeta Platforms Inc.\nGoogle LLC\nAlphabet Inc.")
    if text:
        names = [n for n in text.split('\n') if n.strip()]
else:
    file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    if file:
        df_in = pd.read_excel(file)
        col = st.selectbox("Select column with company names", df_in.columns)
        if col:
            names = df_in[col].dropna().astype(str).tolist()

if st.button("Run Mapper", type="primary") and names:
    df_mapped = map_companies(names)
    st.session_state["mapped_df"] = df_mapped
    
if "mapped_df" in st.session_state:
    st.divider()
    st.subheader("Mapping Results")
    st.caption("Review the suggested canonical names. Edit the `canonical_name` column as needed before downloading.")
    
    # Make a copy to avoid mutating session state implicitly if not using data_editor key
    df_display = st.session_state["mapped_df"].copy()
    
    # Render editable dataframe
    edited_df = st.data_editor(
        df_display, 
        use_container_width=True,
        column_config={
            "original_name": st.column_config.TextColumn("Original Name", disabled=True),
            "canonical_name": st.column_config.TextColumn("Canonical Name", disabled=False),
            "score": st.column_config.NumberColumn("Score", disabled=True),
            "status": st.column_config.TextColumn("Status", disabled=True)
        },
        hide_index=True
    )
    
    # Download button
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        edited_df.to_excel(writer, index=False)
    
    st.download_button(
        label="📥 Download Mapping to Excel",
        data=buf.getvalue(),
        file_name="company_mapping.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
