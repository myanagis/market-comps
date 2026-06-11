import streamlit as st
import pandas as pd
from market_comps.db.session import get_db
from market_comps.db.models import Organization, ProgramProfile, ProgramCohort

st.set_page_config(page_title="Programs & Cohorts", page_icon="🚀", layout="wide")
st.title("🚀 Programs & Cohorts")
st.markdown("Manage accelerator programs, grants, incubators, and their cohorts.")

try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

tab_dir, tab_add_prog, tab_add_cohort = st.tabs(["Directory", "Add Program", "Add Cohort"])

with tab_dir:
    st.subheader("All Programs")
    programs = db.query(ProgramProfile).all()
    if not programs:
        st.info("No programs found.")
    else:
        prog_data = []
        for p in programs:
            prog_data.append({
                "Program ID": p.id,
                "Organization": p.parent_organization.name if p.parent_organization else "Unknown",
                "Program Name": p.program_name,
                "Type": p.program_type or "",
                "Status": p.status or "",
                "Start Date": p.start_date.strftime("%Y-%m-%d") if p.start_date else "",
                "End Date": p.end_date.strftime("%Y-%m-%d") if p.end_date else ""
            })
        st.dataframe(pd.DataFrame(prog_data), use_container_width=True, hide_index=True)
        
    st.divider()
    st.subheader("All Cohorts")
    cohorts = db.query(ProgramCohort).all()
    if not cohorts:
        st.info("No cohorts found.")
    else:
        cohort_data = []
        for c in cohorts:
            cohort_data.append({
                "Cohort ID": c.id,
                "Program": c.program.program_name if c.program else "Unknown",
                "Cohort Name": c.cohort_name,
                "Start Date": c.start_date.strftime("%Y-%m-%d") if c.start_date else "",
                "End Date": c.end_date.strftime("%Y-%m-%d") if c.end_date else ""
            })
        st.dataframe(pd.DataFrame(cohort_data), use_container_width=True, hide_index=True)

with tab_add_prog:
    orgs = db.query(Organization).order_by(Organization.name).all()
    org_options = {org.id: f"{org.name} ({org.primary_domain})" for org in orgs}
    
    if not org_options:
        st.warning("You must create an Organization first before creating a Program.")
    else:
        with st.form("program_form", clear_on_submit=True):
            parent_id = st.selectbox("Parent Organization *", options=list(org_options.keys()), format_func=lambda x: org_options[x])
            
            col1, col2 = st.columns(2)
            name = col1.text_input("Program Name *")
            p_type = col2.text_input("Program Type (e.g. ACCELERATOR, GRANT)")
            
            desc = st.text_area("Description")
            
            submitted = st.form_submit_button("Create Program")
            if submitted:
                if not name:
                    st.error("Program Name is required.")
                else:
                    try:
                        record = ProgramProfile(parent_organization_id=parent_id, program_name=name, program_type=p_type, description=desc)
                        db.add(record)
                        db.commit()
                        st.success(f"Successfully created Program: {name}!")
                    except Exception as e:
                        db.rollback()
                        st.error(f"Error saving to DB: {str(e)}")

with tab_add_cohort:
    programs = db.query(ProgramProfile).all()
    if not programs:
        st.warning("You must create a Program first before creating a Cohort.")
    else:
        prog_options = {p.id: f"{p.parent_organization.name if p.parent_organization else ''} — {p.program_name}" for p in programs}
        with st.form("cohort_form", clear_on_submit=True):
            program_id = st.selectbox("Parent Program *", options=list(prog_options.keys()), format_func=lambda x: prog_options[x])
            
            col1, col2 = st.columns(2)
            cohort_name = col1.text_input("Cohort Name * (e.g. Cohort 6, Winter 2025)")
            desc = st.text_area("Description")
            
            submitted = st.form_submit_button("Create Cohort")
            if submitted:
                if not cohort_name:
                    st.error("Cohort Name is required.")
                else:
                    try:
                        cohort = ProgramCohort(
                            program_id=program_id,
                            cohort_name=cohort_name,
                            description=desc
                        )
                        db.add(cohort)
                        db.commit()
                        st.success(f"Successfully created cohort: {cohort_name}!")
                    except Exception as e:
                        db.rollback()
                        st.error(f"Error saving to DB: {str(e)}")
