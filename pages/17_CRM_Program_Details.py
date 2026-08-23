import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from market_comps.db.session import get_db
from market_comps.db.models import Organization, ProgramProfile, ProgramCohort

st.set_page_config(page_title="Program Details", page_icon="🚀", layout="wide")
st.title("🚀 Program Details")

try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

def display_program_details(program_id):
    prog = db.query(ProgramProfile).options(
        joinedload(ProgramProfile.parent_organization),
        joinedload(ProgramProfile.cohorts)
    ).filter(ProgramProfile.id == int(program_id)).first()

    if not prog:
        st.error(f"Program with ID {program_id} not found.")
        return

    with st.container(border=True):
        st.subheader(f"🚀 {prog.program_name}")
        
        st.markdown("#### Program Information")
        st.write(f"**Parent Organization:** {prog.parent_organization.name if prog.parent_organization else 'N/A'}")
        st.write(f"**Program Type:** {prog.program_type or 'N/A'}")
        st.write(f"**Status:** {prog.status or 'N/A'}")
        
        if prog.start_date:
            st.write(f"**Timeline:** {prog.start_date.strftime('%Y-%m-%d')} to {prog.end_date.strftime('%Y-%m-%d') if prog.end_date else 'Present'}")
            
        if prog.description:
            st.info(prog.description)
            
        st.divider()
        st.subheader("📦 Cohorts")
        if prog.cohorts:
            for c in prog.cohorts:
                with st.expander(f"📦 {c.cohort_name}"):
                    if c.start_date:
                        st.write(f"**Timeline:** {c.start_date.strftime('%Y-%m-%d')} to {c.end_date.strftime('%Y-%m-%d') if c.end_date else 'Present'}")
                    if c.description:
                        st.write(c.description)
        else:
            st.info("No cohorts found for this program.")


program_id = st.query_params.get("id")

col_back, _ = st.columns([1, 5])
with col_back:
    st.page_link("pages/14_CRM_Directory.py", label="← Back to Directory")

if program_id:
    display_program_details(program_id)
else:
    st.info("No Program ID provided.")
