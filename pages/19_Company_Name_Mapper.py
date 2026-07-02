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

def get_meaningful_tokens(name: str) -> list[str]:
    tokens = [t for t in name.upper().split() if t]
    generic = {
        "VENTURES", "HEALTH", "TECH", "TECHNOLOGY", "AI", "LABS", 
        "SYSTEMS", "SOLUTIONS", "GROUP", "HOLDINGS", "CAPITAL", 
        "PARTNERS", "MEDICAL", "SOFTWARE", "ANALYTICS", "ENTERPRISES", 
        "BIOSCIENCES", "INC", "LLC", "CORP", "CORPORATION", "LTD", 
        "LIMITED", "CO", "COMPANY"
    }
    res = []
    for t in tokens:
        if t in generic or t.endswith("MICS"):
            continue
        res.append(t)
    return res

def compare_names(norm_orig: str, norm_can: str) -> tuple[float, str, str]:
    if norm_orig == norm_can:
        return 100.0, "auto_mapped", "Exact normalized match"
        
    meaningful_orig = get_meaningful_tokens(norm_orig)
    meaningful_can = get_meaningful_tokens(norm_can)
    
    score_set = fuzz.token_set_ratio(norm_orig, norm_can)
    score_sort = fuzz.token_sort_ratio(norm_orig, norm_can)
    score_part = fuzz.partial_ratio(norm_orig, norm_can)
    
    # Use sort ratio mostly, but give partial credit for containment
    base_score = max(score_sort, (score_set + score_part) / 2)
    
    if not meaningful_orig or not meaningful_can:
        if base_score >= 92:
            return base_score, "review", "High score but lacks meaningful tokens"
        return base_score, "new", "Name only contains generic words"
            
    orig_contains_can = norm_can in norm_orig and len(meaningful_can) >= 2
    can_contains_orig = norm_orig in norm_can and len(meaningful_orig) >= 2
    
    first_match = (meaningful_orig[0] == meaningful_can[0])
    
    shared = set(meaningful_orig).intersection(set(meaningful_can))
    num_shared = len(shared)
    
    distinct_orig = set(meaningful_orig) - shared
    distinct_can = set(meaningful_can) - shared
    
    avoid_reasons = []
    if not first_match:
        avoid_reasons.append("First meaningful tokens differ")
    if distinct_orig and distinct_can:
        avoid_reasons.append("Both have different distinctive tokens")
    if num_shared == 1 and not distinct_orig and not distinct_can:
        pass # single token exact match
        
    strong_match_reason = None
    if orig_contains_can or can_contains_orig:
        strong_match_reason = "Containment match with >= 2 meaningful tokens"
    elif first_match and base_score >= 92:
        strong_match_reason = "First meaningful token matches and score >= 92"
    elif num_shared >= 2 and base_score >= 90:
        strong_match_reason = ">= 2 meaningful tokens overlap and score >= 90"
        
    if num_shared == 0:
        return base_score, "new", "No/low overlap with prior companies"
        
    if avoid_reasons:
        if base_score >= 82:
            return base_score, "new" if len(avoid_reasons) >= 2 else "review", f"Avoided auto-match: {', '.join(avoid_reasons)}"
        return base_score, "new", f"Different entities ({', '.join(avoid_reasons)})"
        
    if strong_match_reason:
        return base_score, "auto_mapped", strong_match_reason
        
    if base_score >= 92:
        return base_score, "auto_mapped", "High score >= 92"
    elif base_score >= 82:
        return base_score, "review", "Score between 82 and 91"
    else:
        return base_score, "new", "Low score < 82"

def map_companies(names: list[str]) -> pd.DataFrame:
    canonical_list = []
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
                "status": "new",
                "reason": "First entry"
            })
            continue
            
        best_match = None
        best_score = -1
        best_status = "new"
        best_reason = ""
        
        for can in canonical_list:
            score, status, reason = compare_names(norm, can["normalized"])
            
            def status_weight(s):
                if s == "auto_mapped": return 3
                if s == "review": return 2
                return 1
                
            if (status_weight(status), score) > (status_weight(best_status), best_score):
                best_score = score
                best_match = can
                best_status = status
                
                if status == "new" and score < 60:
                    best_reason = "No strong match found"
                elif status == "new" and "No/low overlap with prior companies" in reason:
                    best_reason = reason
                else:
                    best_reason = f"{reason} (vs '{can['canonical']}')"
                
        if best_status in ("auto_mapped", "review"):
            results.append({
                "original_name": orig_str,
                "canonical_name": best_match["canonical"],
                "score": round(best_score, 1),
                "status": best_status,
                "reason": best_reason
            })
        else:
            canonical_list.append({"canonical": orig_str, "normalized": norm})
            results.append({
                "original_name": orig_str,
                "canonical_name": orig_str,
                "score": round(best_score, 1) if best_match else 0.0,
                "status": "new",
                "reason": best_reason or "No strong match found"
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
            "canonical_name": st.column_config.TextColumn("Mapped Name", disabled=False),
            "score": st.column_config.NumberColumn("Matching Score", disabled=True),
            "status": st.column_config.TextColumn("Status", disabled=True),
            "reason": st.column_config.TextColumn("Reason", disabled=True)
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
