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

# 5. Render Premium Login Interface if not authenticated
if not st.session_state["authenticated"]:
    # Custom premium design injection
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Font and theme adjustments */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(circle at 50% 50%, #151a2e 0%, #0b0d13 100%);
        }
        
        /* Center container and Card style */
        .login-card-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 40px 10px;
        }
        
        .login-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 48px;
            width: 100%;
            max-width: 480px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
            text-align: center;
            animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(24px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .portal-logo {
            font-size: 48px;
            margin-bottom: 12px;
            animation: float 4s ease-in-out infinite;
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .portal-title {
            font-size: 32px;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        
        .portal-subtitle {
            font-size: 15px;
            color: #9ca3af;
            margin-bottom: 32px;
        }
        
        /* Styled tab headers matching our color scheme */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            justify-content: center;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            background-color: rgba(255, 255, 255, 0.02);
            color: #9ca3af;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.2s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(99, 102, 241, 0.15) !important;
            color: #a5b4fc !important;
            border-color: rgba(99, 102, 241, 0.4) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="login-card-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div class="login-card">
                <div class="portal-logo">📊</div>
                <div class="portal-title">Market Comps</div>
                <div class="portal-subtitle">Secure Administration Portal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        tab_google, tab_dev = st.tabs(["🔒 Supabase OAuth", "🛠️ Developer Bypass"])
        
        with tab_google:
            anon_key = get_supabase_anon_key()
            project_ref = get_supabase_project_ref()
            
            if not anon_key or not project_ref:
                st.warning("⚠️ Supabase Credentials missing. Please configure `SUPABASE_ANON_KEY` in `.streamlit/secrets.toml` to enable Google Auth.")
                st.info("💡 You can use the 'Developer Bypass' tab to sign in and test locally.")
            else:
                # Dynamic redirect to current app base URL
                # st.context.url doesn't include query params, which is perfect as callback destination
                try:
                    app_base_url = st.context.url
                except Exception:
                    app_base_url = "http://localhost:8501/"
                    
                google_url = get_google_auth_url(app_base_url)
                
                st.markdown(
                    f"""
                    <div style="margin: 24px 0;">
                        <a href="{google_url}" target="_self" style="text-decoration: none;">
                            <button style="
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                width: 100%;
                                background-color: #ffffff;
                                color: #1f2937;
                                border: 1px solid #e5e7eb;
                                border-radius: 12px;
                                padding: 12px 24px;
                                font-size: 16px;
                                font-weight: 600;
                                cursor: pointer;
                                transition: all 0.2s ease;
                                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
                            "
                            onmouseover="this.style.backgroundColor='#f9fafb'; this.style.transform='translateY(-1px)';"
                            onmouseout="this.style.backgroundColor='#ffffff'; this.style.transform='translateY(0)';"
                            >
                                <svg style="width: 20px; height: 20px; margin-right: 12px;" viewBox="0 0 24 24">
                                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
                                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
                                </svg>
                                Sign in with Google
                            </button>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        with tab_dev:
            st.markdown("<p style='color: #9ca3af; font-size: 14px; text-align: center; margin-bottom: 16px;'>Simulate a successful login for local development and testing.</p>", unsafe_allow_html=True)
            dev_email = st.text_input("Developer Email", value="admin@marketcomps.dev", placeholder="e.g. admin@company.com")
            
            if st.button("🚀 Launch Developer Session", use_container_width=True):
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = dev_email
                st.success("Welcome back! Redirecting...")
                st.rerun()
                
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# 6. Authenticated User - Sidebar Log Out Option
with st.sidebar:
    st.markdown(
        f"""
        <div style="background-color: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); padding: 12px; border-radius: 12px; margin-bottom: 20px;">
            <div style="font-size: 12px; color: #a5b4fc;">Authenticated User</div>
            <div style="font-size: 14px; font-weight: 600; color: #ffffff; word-break: break-all;">👤 {st.session_state['user_email']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["user_email"] = None
        st.success("Logged out successfully!")
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

