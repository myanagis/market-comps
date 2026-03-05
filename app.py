# app.py
"""
Market Comps — Entry point with explicit page navigation.

Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Market Comps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("pages/1_Public_Comps.py",      title="Public Comps",           icon="📊"),
    st.Page("pages/2_Competition_Finder.py",title="Competition Finder",     icon="🏢"),
    st.Page("pages/2_PDF_Parser.py",        title="PDF Parser",             icon="📄"),
    st.Page("pages/3_LLM_Cross_Checker.py", title="Chorus of LLMs",         icon="🎼"),
    st.Page("pages/4_Company_Primer.py",    title="Company Primer",         icon="📚"),
    st.Page("pages/5_FAQ.py",               title="FAQ",                    icon="❓"),
    st.Page("pages/6_Waterfall_Calculator.py", title="Waterfall Calculator",icon="💧"),
])

pg.run()
