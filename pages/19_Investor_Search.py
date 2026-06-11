import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import or_
from market_comps.db.session import get_db
from market_comps.db.models import Organization, InvestorProfile

st.set_page_config(page_title="Investor Search", page_icon="🔍", layout="wide")
st.title("🔍 Investor Search")
st.markdown("Search and filter for investors based on themes, location, and other criteria.")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Helper to get all unique themes
@st.cache_data(ttl=60)
def get_all_investor_themes():
    # We must use a new session here or pass db to function, but st.cache_data requires hashable inputs.
    # It's better to fetch inline without caching for simplicity, or cache without the db arg.
    pass

def fetch_themes_and_locations():
    profiles = db.query(InvestorProfile.themes).filter(InvestorProfile.themes.is_not(None)).all()
    themes = set()
    for (t,) in profiles:
        if t and isinstance(t, list):
            themes.update(t)
            
    # Fetch states
    orgs = db.query(Organization.state, Organization.country).filter(Organization.organization_type == "INVESTOR").all()
    states = set()
    countries = set()
    for s, c in orgs:
        if s: states.add(s)
        if c: countries.add(c)
        
    return sorted(list(themes)), sorted(list(states)), sorted(list(countries))

themes_list, states_list, countries_list = fetch_themes_and_locations()

with st.sidebar:
    st.header("Filters")
    search_query = st.text_input("Text Search", placeholder="Name, domain, or notes...")
    
    selected_themes = st.multiselect("Filter by Theme", options=themes_list)
    
    selected_states = st.multiselect("Filter by State", options=states_list)
    selected_countries = st.multiselect("Filter by Country", options=countries_list)
    
    # We could also add Investor Type / Stage
    inv_types = [r[0] for r in db.query(InvestorProfile.investor_type).distinct().all() if r[0]]
    selected_types = st.multiselect("Investor Type", options=sorted(inv_types))

# Build Query
q = db.query(Organization).options(
    joinedload(Organization.investor_profile)
).filter(Organization.organization_type == "INVESTOR")

if search_query:
    search_filter = f"%{search_query}%"
    q = q.filter(
        or_(
            Organization.name.ilike(search_filter),
            Organization.primary_domain.ilike(search_filter),
            Organization.description.ilike(search_filter),
            InvestorProfile.user_notes.ilike(search_filter)
        )
    )

if selected_states:
    q = q.filter(Organization.state.in_(selected_states))

if selected_countries:
    q = q.filter(Organization.country.in_(selected_countries))

if selected_types:
    # Need to join InvestorProfile if not already joined in filter context
    # joinedload doesn't automatically join for filtering
    pass 

# Actually for filtering on joined columns, we need a real join:
q = db.query(Organization).join(
    InvestorProfile, Organization.id == InvestorProfile.organization_id, isouter=True
).options(
    joinedload(Organization.investor_profile)
).filter(Organization.organization_type == "INVESTOR")

if search_query:
    search_filter = f"%{search_query}%"
    q = q.filter(
        or_(
            Organization.name.ilike(search_filter),
            Organization.primary_domain.ilike(search_filter),
            Organization.description.ilike(search_filter),
            InvestorProfile.user_notes.ilike(search_filter)
        )
    )

if selected_states:
    q = q.filter(Organization.state.in_(selected_states))

if selected_countries:
    q = q.filter(Organization.country.in_(selected_countries))

if selected_types:
    q = q.filter(InvestorProfile.investor_type.in_(selected_types))

# JSON filtering in SQLite/Postgres can be tricky with SQLAlchemy without dialects.
# We'll filter themes in-memory since the CRM dataset is usually small enough for this.
results = q.order_by(Organization.name).all()

# In-memory filter for themes
if selected_themes:
    filtered_results = []
    for org in results:
        prof = org.investor_profile
        if prof and prof.themes:
            # Check if there is an intersection
            if set(selected_themes).intersection(set(prof.themes)):
                filtered_results.append(org)
    results = filtered_results

st.write(f"### Results: {len(results)} investors found")

if results:
    data = []
    for o in results:
        prof = o.investor_profile
        
        row = {
            "ID": o.id,
            "Name": o.name,
            "Domain": o.primary_domain,
            "City": o.city or "",
            "State": o.state or "",
            "Country": o.country or "",
            "Type": prof.investor_type if prof else "",
            "Pref Stage": prof.preferred_stage if prof else "",
            "Founded": prof.founded_year if prof else "",
            "Themes": ", ".join(prof.themes) if prof and prof.themes else "",
            "Description": o.description or "",
            "User Notes": prof.user_notes if prof else "",
            "Status": o.status or ""
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": None # Hide the DB ID
        }
    )
else:
    st.info("No investors match your criteria.")
