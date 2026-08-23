import streamlit as st
import pandas as pd
from market_comps.db.session import get_db
from market_comps.db.models import (
    Organization, FundProfile, ProgramProfile, PersonOrganizationRole, 
    ProgramMembership, Pipeline, MetricObservation, FinancingRound, 
    RoundInvestor, Transaction, EntityMatch, AuditTrail, PersonEmail
)

st.set_page_config(page_title="Data Cleanup & Merge", page_icon="🧹", layout="wide")
st.title("🧹 Data Cleanup & Merge")
st.markdown("Safely merge duplicate organizations. All related records (funds, roles, metrics, documents) will be re-parented to the Primary Organization.")

try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Helper function to perform merge
def merge_organizations(primary_id, duplicate_id):
    if primary_id == duplicate_id:
        raise ValueError("Primary and Duplicate cannot be the same organization.")
        
    primary = db.query(Organization).get(primary_id)
    duplicate = db.query(Organization).get(duplicate_id)
    
    if not primary or not duplicate:
        raise ValueError("One or both organizations not found.")
        
    # Re-parent funds
    db.query(FundProfile).filter(FundProfile.parent_organization_id == duplicate_id).update({"parent_organization_id": primary_id})
    # Re-parent programs
    db.query(ProgramProfile).filter(ProgramProfile.parent_organization_id == duplicate_id).update({"parent_organization_id": primary_id})
    # Re-parent roles
    db.query(PersonOrganizationRole).filter(PersonOrganizationRole.organization_id == duplicate_id).update({"organization_id": primary_id})
    # Re-parent program memberships
    db.query(ProgramMembership).filter(ProgramMembership.company_organization_id == duplicate_id).update({"company_organization_id": primary_id})
    # Re-parent pipelines
    db.query(Pipeline).filter(Pipeline.organization_id == duplicate_id).update({"organization_id": primary_id})
    # Re-parent metric observations
    db.query(MetricObservation).filter(MetricObservation.company_id == duplicate_id).update({"company_id": primary_id})
    # Re-parent financing rounds
    db.query(FinancingRound).filter(FinancingRound.company_id == duplicate_id).update({"company_id": primary_id})
    # Re-parent investments made
    db.query(RoundInvestor).filter(RoundInvestor.investor_id == duplicate_id).update({"investor_id": primary_id})
    # Re-parent transactions
    db.query(Transaction).filter(Transaction.target_company_id == duplicate_id).update({"target_company_id": primary_id})
    db.query(Transaction).filter(Transaction.acquirer_company_id == duplicate_id).update({"acquirer_company_id": primary_id})
    # Re-parent person emails
    db.query(PersonEmail).filter(PersonEmail.organization_id == duplicate_id).update({"organization_id": primary_id})
    
    # Re-parent EntityMatch and AuditTrail (stored as strings)
    db.query(EntityMatch).filter(
        EntityMatch.canonical_entity_type == "Organization",
        EntityMatch.canonical_entity_id == str(duplicate_id)
    ).update({"canonical_entity_id": str(primary_id)})
    
    db.query(AuditTrail).filter(
        AuditTrail.canonical_entity_type == "ORGANIZATION",
        AuditTrail.canonical_entity_id == str(duplicate_id)
    ).update({"canonical_entity_id": str(primary_id)})
    
    # Delete the duplicate organization
    db.delete(duplicate)
    db.commit()

# --- UI ---

st.subheader("Merge Organizations")
orgs = db.query(Organization).order_by(Organization.name).all()
org_options = {org.id: f"{org.name} (Type: {org.organization_type}, Domain: {org.primary_domain or 'None'}) - ID: {org.id}" for org in orgs}

with st.form("merge_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🥇 Primary Organization (Keep)")
        st.write("This organization will be retained. All relationships will point here.")
        primary_id = st.selectbox("Select Primary", options=list(org_options.keys()), format_func=lambda x: org_options[x], key="primary")
        
    with col2:
        st.write("### 🗑️ Duplicate Organization (Delete)")
        st.write("This organization will be **deleted** after its relationships are transferred.")
        duplicate_id = st.selectbox("Select Duplicate", options=list(org_options.keys()), format_func=lambda x: org_options[x], key="duplicate")
        
    st.warning("⚠️ **WARNING:** This action is irreversible. Double-check your selections.")
    
    submitted = st.form_submit_button("Merge Organizations")
    
    if submitted:
        if primary_id == duplicate_id:
            st.error("Please select two different organizations.")
        else:
            try:
                merge_organizations(primary_id, duplicate_id)
                st.success(f"Successfully merged organization ID {duplicate_id} into {primary_id}!")
                st.rerun()
            except Exception as e:
                db.rollback()
                st.error(f"Merge failed: {e}")
