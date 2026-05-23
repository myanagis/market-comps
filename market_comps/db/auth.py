import re
import requests
import streamlit as st
from market_comps.db.session import get_database_url

def get_supabase_project_ref() -> str:
    """Extract project ref (e.g. yylenmafzuulfgpxfamm) from the SUPABASE_URL db string."""
    db_url = get_database_url(direct=False)
    if not db_url:
        return ""
    # Look for postgres.project_ref:password in the URL
    match = re.search(r'postgres\.([a-z0-9]+):', db_url)
    if match:
        return match.group(1)
    return ""

def get_supabase_anon_key() -> str:
    """Retrieve the Supabase anon key from st.secrets."""
    try:
        return st.secrets.get("SUPABASE_ANON_KEY", "")
    except Exception:
        return ""

def get_google_auth_url(redirect_url: str) -> str:
    """Generate the Supabase Google Auth redirect URL."""
    project_ref = get_supabase_project_ref()
    if not project_ref:
        return ""
    # We use provider=google and set the redirect_to parameter
    return f"https://{project_ref}.supabase.co/auth/v1/authorize?provider=google&redirect_to={redirect_url}"

def verify_supabase_token(token: str) -> dict:
    """Verify access token against Supabase API and return user profile details if valid."""
    project_ref = get_supabase_project_ref()
    anon_key = get_supabase_anon_key()
    
    if not project_ref or not anon_key:
        # Fallback for debugging if keys are missing
        return {}
        
    url = f"https://{project_ref}.supabase.co/auth/v1/user"
    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.warning(f"Error calling Supabase Auth API: {e}")
        
    return {}
