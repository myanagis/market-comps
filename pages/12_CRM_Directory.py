import streamlit as st
from market_comps.db.session import get_db
from market_comps.db.models import Organization, Person, EntityAuditTrail
from sqlalchemy.orm import joinedload
from sqlalchemy import or_
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, ColumnsAutoSizeMode

st.set_page_config(page_title="CRM Directory", page_icon="🗂️", layout="wide")
st.title("🗂️ CRM Directory")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed. Did you configure secrets.toml? Error: {e}")
    st.stop()

def get_org_df(org_type, search_text):
    q = db.query(Organization).options(
        joinedload(Organization.company_profile),
        joinedload(Organization.investor_profile),
    ).filter(Organization.organization_type == org_type)
    
    if search_text:
        search_filter = f"%{search_text}%"
        q = q.filter(
            or_(
                Organization.name.ilike(search_filter),
                Organization.normalized_name.ilike(search_filter),
                Organization.primary_domain.ilike(search_filter)
            )
        )
    orgs = q.order_by(Organization.created_at.desc()).limit(200).all()
    
    data = []
    for o in orgs:
        row = {
            "ID": o.id,
            "Name": o.name,
            "Domain": o.primary_domain,
            "Website": o.website_url,
            "City": o.city,
            "Status": o.status,
            "Created": o.created_at.strftime("%Y-%m-%d") if o.created_at else ""
        }
        if o.company_profile:
            row["Industry"] = o.company_profile.industry
            row["Stage"] = o.company_profile.company_stage
        if o.investor_profile:
            row["Inv Type"] = o.investor_profile.investor_type
        data.append(row)
    
    return pd.DataFrame(data)

def render_aggrid(df, key, record_type="ORGANIZATION"):
    if df.empty:
        st.info("No records found.")
        return
    
    gb = GridOptionsBuilder.from_dataframe(df)
    # Enable single row selection - clicking anywhere in the row works
    gb.configure_selection(selection_mode='single', use_checkbox=True)
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False)
    
    # Hide ID
    gb.configure_column("ID", hide=True)
    
    grid_options = gb.build()
    
    st.write("👆 *Click a row to select it, then click 'View Details' below.*")
    
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        theme='streamlit',
        key=key
    )
    
    selected_rows = grid_response['selected_rows']
    
    # Selection logic handling different st-aggrid versions
    has_selection = False
    selected_id = None
    selected_name = ""
    
    if selected_rows is not None:
        if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
            row = selected_rows.iloc[0]
            selected_id = row["ID"]
            selected_name = row.get("Name", "Selected Record")
            has_selection = True
        elif isinstance(selected_rows, list) and len(selected_rows) > 0:
            row = selected_rows[0]
            selected_id = row.get("ID")
            selected_name = row.get("Name", "Selected Record")
            has_selection = True

    if has_selection:
        if st.button(f"🔍 View Full Details for {selected_name}", key=f"view_{key}", type="primary"):
            st.query_params["type"] = record_type
            st.query_params["id"] = str(selected_id)
            st.switch_page("pages/15_Record_Detail.py")
    else:
        st.button("🔍 Select a record to view details", key=f"view_{key}_disabled", disabled=True)

tab_companies, tab_investors, tab_people = st.tabs(["🏢 Companies", "🏦 Investors", "👤 People"])

# --- COMPANIES TAB ---
with tab_companies:
    c_search = st.text_input("Search Companies...", placeholder="e.g. Acme Corp", key="search_companies")
    df_companies = get_org_df("COMPANY", c_search)
    render_aggrid(df_companies, "grid_companies", "ORGANIZATION")

# --- INVESTORS TAB ---
with tab_investors:
    i_search = st.text_input("Search Investors...", placeholder="e.g. Sequoia", key="search_investors")
    df_investors = get_org_df("INVESTOR", i_search)
    render_aggrid(df_investors, "grid_investors", "ORGANIZATION")

# --- PEOPLE TAB ---
with tab_people:
    p_search = st.text_input("Search People...", placeholder="e.g. Jane Doe", key="search_people")
    
    q = db.query(Person)
    if p_search:
        search_filter = f"%{p_search}%"
        q = q.filter(
            or_(
                Person.full_name.ilike(search_filter),
                Person.first_name.ilike(search_filter),
                Person.last_name.ilike(search_filter)
            )
        )
    people = q.order_by(Person.created_at.desc()).limit(200).all()
    
    data = []
    for p in people:
        data.append({
            "ID": p.id,
            "Name": p.full_name or f"{p.first_name} {p.last_name}",
            "LinkedIn": p.linkedin_url,
            "City": p.city,
            "State": p.state,
            "Created": p.created_at.strftime("%Y-%m-%d") if p.created_at else ""
        })
    df_people = pd.DataFrame(data)
    render_aggrid(df_people, "grid_people", "PERSON")

