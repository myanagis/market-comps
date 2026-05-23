# app.py
"""
Market Comps — Entry point with explicit page navigation and Supabase Auth.

Run with:
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
from market_comps.db.auth import (
    get_supabase_project_ref,
    get_supabase_anon_key,
    get_google_auth_url,
    verify_supabase_token
)

# 1. Page Configuration
st.set_page_config(
    page_title="Market Comps Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Session State Initialization
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

# 3. Handle OAuth Redirect Callbacks (Query Parameters)
query_params = st.query_params
if "access_token" in query_params:
    token = query_params["access_token"]
    # Verify the token against Supabase API
    user_info = verify_supabase_token(token)
    if user_info and "email" in user_info:
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = user_info["email"]
    else:
        # Fallback if verification API fails or is not configured but token was returned
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = "google-user@supabase.io"
    
    # Clear parameters and rerun to clean up URL
    st.query_params.clear()
    st.rerun()

# 4. Inject JavaScript to convert URL Hash fragment into Query params
components.html(
    """
    <script>
    try {
        const parentHash = window.parent.location.hash;
        if (parentHash && parentHash.includes('access_token')) {
            const hash = parentHash.substring(1);
            const params = new URLSearchParams(hash);
            const accessToken = params.get('access_token');
            if (accessToken) {
                // Redirect parent window to clean URL with query parameter
                window.parent.location.href = window.parent.location.origin + window.parent.location.pathname + '?access_token=' + encodeURIComponent(accessToken);
            }
        }
    } catch (e) {
        console.error("OAuth redirect parse error:", e);
    }
    </script>
    """,
    height=0,
    width=0,
)

# 5. Sidebar Authentication Widget (Optional Login)
with st.sidebar:
    st.markdown("---")
    if st.session_state["authenticated"]:
        st.success(f"👤 {st.session_state['user_email']}")
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user_email"] = None
            st.success("Logged out successfully!")
            st.rerun()
    else:
        with st.expander("🔑 Admin Login", expanded=False):
            tab_google, tab_dev = st.tabs(["Google OAuth", "Dev Bypass"])
            
            with tab_google:
                anon_key = get_supabase_anon_key()
                project_ref = get_supabase_project_ref()
                
                if not anon_key or not project_ref:
                    st.warning("Please configure `SUPABASE_ANON_KEY` in `secrets.toml` to enable Google Auth.")
                else:
                    try:
                        app_base_url = st.context.url
                    except Exception:
                        app_base_url = "http://localhost:8501/"
                        
                    google_url = get_google_auth_url(app_base_url)
                    
                    st.markdown(
                        f"""
                        <div style="margin: 10px 0; text-align: center;">
                            <a href="{google_url}" target="_self">
                                <button style="
                                    cursor: pointer;
                                    padding: 8px 16px;
                                    border: 1px solid #ccc;
                                    border-radius: 6px;
                                    background-color: #ffffff;
                                    color: #000000;
                                    font-size: 14px;
                                    font-weight: bold;
                                    width: 100%;
                                ">
                                    Sign in with Google
                                </button>
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            with tab_dev:
                dev_email = st.text_input("Developer Email", value="admin@marketcomps.dev")
                if st.button("🚀 Sign In (Bypass)", use_container_width=True):
                    st.session_state["authenticated"] = True
                    st.session_state["user_email"] = dev_email
                    st.success("Welcome back!")
                    st.rerun()

# 7. Navigation Setup
pg = st.navigation({
    "Apps": [
        st.Page("pages/11_CT_Business_Registry.py", title="CT Business Registry", icon="🏢"),
    ],
    "CRM": [
        st.Page("pages/12_CRM_Directory.py", title="CRM Directory", icon="🗂️"),
        st.Page("pages/13_Data_Ingestion.py", title="Data Ingestion", icon="📡"),
        st.Page("pages/14_CRM_Record_Manager.py", title="CRM Record Manager", icon="📝"),
        st.Page("pages/15_Record_Detail.py", title="Record Details", icon="📝"),
    ],
    "Tools (Beta)": [
        st.Page("pages/1_Public_Comps.py",      title="Public Comps",           icon="📊"),
        st.Page("pages/2_Competition_Finder.py",title="Competition Finder",     icon="🏢"),
        st.Page("pages/2_Data_Extraction.py",   title="Data Extraction",        icon="📄"),
        st.Page("pages/3_LLM_Cross_Checker.py", title="Chorus of LLMs",         icon="🎼"),
        st.Page("pages/4_Company_Primer.py",    title="Company Primer",         icon="📚"),
        st.Page("pages/5_FAQ.py",               title="FAQ",                    icon="❓"),
        st.Page("pages/6_Waterfall_Calculator.py", title="Waterfall Calculator",icon="💧"),
        st.Page("pages/7_Directory_Analyzer.py", title="Directory Analyzer",    icon="📁"),
        st.Page("pages/8_Cash_Flow_Analysis.py", title="Cash Flow Analysis",    icon="💸"),
        st.Page("pages/9_Portfolio_Company_Analysis.py", title="Portfolio Company Analysis", icon="🏢"),
        st.Page("pages/10_Schema_Driven_Framework.py", title="Schema-Driven Framework", icon="📋"),
    ],
    "Admin": [
        st.Page("pages/16_Admin.py", title="Admin DB Manager", icon="🔒"),
    ]
})

pg.run()

