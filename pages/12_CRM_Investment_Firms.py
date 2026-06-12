import streamlit as st
import pandas as pd
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, and_, desc
import json
from market_comps.db.session import get_db
from market_comps.utils import format_est_datetime, format_currency
from market_comps.db.models import (
    Organization, Person, InvestorProfile, FundProfile,
    ProgramMembership, PersonOrganizationRole, AuditTrail,
    ProgramProfile, ProgramCohort, EntityMatch, ExtractedEntity, ExtractionJob, PipelineRun, DocumentText, SourceDocument
)
from market_comps.ingestion.reconciler import log_mutation

st.set_page_config(page_title="Investment Firms Directory", page_icon="🏦", layout="wide")

st.title("🏦 Investment Firms Directory")
st.markdown("Browse and search investment firms and their funds in the CRM.")

tab_firms, tab_funds, tab_add_firm, tab_add_fund = st.tabs(["Investment Firms", "Fund Profiles", "Add Investment Firm", "Add Fund"])

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

def get_all_investor_themes(db):
    profiles = db.query(InvestorProfile.themes).filter(InvestorProfile.themes.is_not(None)).all()
    themes = set()
    for (t,) in profiles:
        if t and isinstance(t, list):
            themes.update(t)
    return sorted(list(themes))

def get_all_fund_themes(db):
    profiles = db.query(FundProfile.themes).filter(FundProfile.themes.is_not(None)).all()
    themes = set()
    for (t,) in profiles:
        if t and isinstance(t, list):
            themes.update(t)
    return sorted(list(themes))

@st.dialog("Edit Investment Firm")
def edit_firm_dialog(org):
    with st.form("edit_firm"):
        st.write(f"Edit details for **{org.name}**")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=org.name or "")
            domain = st.text_input("Domain", value=org.primary_domain or "")
            website = st.text_input("Website URL", value=org.website_url or "")
            linkedin = st.text_input("LinkedIn URL", value=org.linkedin_url or "")
        with col2:
            city = st.text_input("City", value=org.city or "")
            state = st.text_input("State", value=org.state or "")
            country = st.text_input("Country", value=org.country or "")
            status_opts = ["ACTIVE", "INACTIVE", "ACQUIRED", "DEFUNCT"]
            status = st.selectbox("Status", status_opts, index=status_opts.index(org.status) if org.status in status_opts else 0)
            
        desc = st.text_area("Description", value=org.description or "")
        
        st.divider()
        st.write("Investor Profile")
        col3, col4 = st.columns(2)
        prof = org.investor_profile
        with col3:
            investor_type = st.text_input("Investor Type", value=prof.investor_type if prof else "")
            founded = st.number_input("Founded Year", value=prof.founded_year if prof and prof.founded_year else None, step=1, placeholder="YYYY")
        with col4:
            preferred_stage = st.text_input("Preferred Stage", value=prof.preferred_stage if prof else "")
            
        user_notes = st.text_area("User Notes", value=prof.user_notes if prof else "")
        all_themes = get_all_investor_themes(db)
        selected_themes = st.multiselect("Themes", options=all_themes, default=prof.themes if prof and prof.themes else [])
        new_themes = st.text_input("Add New Themes (comma separated)")
            
        if st.form_submit_button("Save Changes"):
            user = st.session_state.get("user_email", "SYSTEM")
            
            def check_and_update(entity_type, entity_id, field_name, old_val, new_val, obj):
                if str(old_val) != str(new_val) and (old_val or new_val):
                    log_mutation(
                        db, entity_type, entity_id, "UPDATE",
                        field_name=field_name,
                        old_value=str(old_val),
                        new_value=str(new_val),
                        source="USER_EDIT",
                        created_by=user
                    )
                    setattr(obj, field_name, new_val)
                    
            check_and_update("ORGANIZATION", org.id, "name", org.name, name, org)
            check_and_update("ORGANIZATION", org.id, "primary_domain", org.primary_domain, domain, org)
            check_and_update("ORGANIZATION", org.id, "website_url", org.website_url, website, org)
            check_and_update("ORGANIZATION", org.id, "linkedin_url", org.linkedin_url, linkedin, org)
            check_and_update("ORGANIZATION", org.id, "city", org.city, city, org)
            check_and_update("ORGANIZATION", org.id, "state", org.state, state, org)
            check_and_update("ORGANIZATION", org.id, "country", org.country, country, org)
            check_and_update("ORGANIZATION", org.id, "status", org.status, status, org)
            check_and_update("ORGANIZATION", org.id, "description", org.description, desc, org)
            
            if not prof:
                prof = InvestorProfile(organization_id=org.id)
                db.add(prof)
                db.flush()
                log_mutation(
                    db, "INVESTOR_PROFILE", prof.id, "CREATE",
                    source="USER_EDIT", created_by=user
                )
                org.investor_profile = prof
                
            check_and_update("INVESTOR_PROFILE", prof.id, "investor_type", prof.investor_type, investor_type, prof)
            check_and_update("INVESTOR_PROFILE", prof.id, "preferred_stage", prof.preferred_stage, preferred_stage, prof)
            check_and_update("INVESTOR_PROFILE", prof.id, "founded_year", prof.founded_year, founded if founded else None, prof)
            check_and_update("INVESTOR_PROFILE", prof.id, "user_notes", prof.user_notes, user_notes, prof)
            
            final_themes = list(selected_themes)
            if new_themes:
                final_themes.extend([t.strip() for t in new_themes.split(",") if t.strip()])
            final_themes = list(set(final_themes))
            check_and_update("INVESTOR_PROFILE", prof.id, "themes", prof.themes, final_themes, prof)
            
            db.commit()
            st.rerun()

@st.dialog("Edit Fund Profile")
def edit_fund_dialog(fund):
    with st.form("edit_fund"):
        st.write(f"Edit details for **{fund.fund_name}**")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Fund Name", value=fund.fund_name)
            vintage = st.number_input("Vintage Year", min_value=1980, max_value=2100, value=fund.vintage_year if fund.vintage_year else None, step=1)
            raised = st.text_input("Size Raised", value=fund.fund_size_raised or "")
            reputation_opts = ["High", "Medium", "Low", "Emerging", ""]
            market_rep = st.selectbox("Market Reputation", reputation_opts, index=reputation_opts.index(fund.market_reputation) if fund.market_reputation in reputation_opts else 4)
        with col2:
            f_type = st.text_input("Fund Type", value=fund.fund_type or "")
            target = st.text_input("Size Target", value=fund.fund_size_target or "")
            status = st.text_input("Status", value=fund.status or "")
            
        desc = st.text_area("Description", value=fund.description or "")
        user_notes = st.text_area("User Notes", value=fund.user_notes or "")
        
        all_themes = get_all_fund_themes(db)
        selected_themes = st.multiselect("Themes", options=all_themes, default=fund.themes if fund.themes else [])
        new_themes = st.text_input("Add New Themes (comma separated)")
        
        if st.form_submit_button("Save Changes"):
            user = st.session_state.get("user_email", "SYSTEM")
            
            def check_and_update(entity_type, entity_id, field_name, old_val, new_val, obj):
                if str(old_val) != str(new_val) and (old_val or new_val):
                    log_mutation(db, entity_type, entity_id, "UPDATE", field_name=field_name, old_value=str(old_val), new_value=str(new_val), source="USER_EDIT", created_by=user)
                    setattr(obj, field_name, new_val)
                    
            check_and_update("FUND_PROFILE", fund.id, "fund_name", fund.fund_name, name, fund)
            check_and_update("FUND_PROFILE", fund.id, "vintage_year", fund.vintage_year, vintage if vintage else None, fund)
            check_and_update("FUND_PROFILE", fund.id, "fund_size_raised", fund.fund_size_raised, raised, fund)
            check_and_update("FUND_PROFILE", fund.id, "fund_size_target", fund.fund_size_target, target, fund)
            check_and_update("FUND_PROFILE", fund.id, "fund_type", fund.fund_type, f_type, fund)
            check_and_update("FUND_PROFILE", fund.id, "status", fund.status, status, fund)
            check_and_update("FUND_PROFILE", fund.id, "description", fund.description, desc, fund)
            check_and_update("FUND_PROFILE", fund.id, "market_reputation", fund.market_reputation, market_rep if market_rep else None, fund)
            check_and_update("FUND_PROFILE", fund.id, "user_notes", fund.user_notes, user_notes, fund)
            
            final_themes = list(selected_themes)
            if new_themes:
                final_themes.extend([t.strip() for t in new_themes.split(",") if t.strip()])
            final_themes = list(set(final_themes))
            check_and_update("FUND_PROFILE", fund.id, "themes", fund.themes, final_themes, fund)
            
            db.commit()
            st.rerun()

# Helper to render investor details below the table
def display_investor_details(investor_id):
    org = db.query(Organization).options(
        joinedload(Organization.investor_profile),
        joinedload(Organization.fund_profiles),
        joinedload(Organization.program_profiles).joinedload(ProgramProfile.cohorts),
        joinedload(Organization.program_memberships).joinedload(ProgramMembership.cohort).joinedload(ProgramCohort.program),
        joinedload(Organization.roles).joinedload(PersonOrganizationRole.person)
    ).filter(Organization.id == int(investor_id)).first()

    if not org:
        st.error(f"Investor with ID {investor_id} not found.")
        return

    with st.container(border=True):
        col_h1, col_h2 = st.columns([5, 1])
        with col_h1:
            st.subheader(f"🏦 {org.name}")
        with col_h2:
            if st.button("✏️ Edit", key=f"edit_org_{org.id}", use_container_width=True):
                edit_firm_dialog(org)
        
        st.markdown("#### Firm Details")
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            domain_str = f"[{org.primary_domain}](https://{org.primary_domain})" if org.primary_domain else "N/A"
            st.write(f"**Domain:** {domain_str}")
            website_str = f"[{org.website_url}]({org.website_url})" if org.website_url else "N/A"
            st.write(f"**Website:** {website_str}")
            st.write(f"**LinkedIn:** {org.linkedin_url or 'N/A'}")
            st.write(f"**Location:** {org.city or ''}, {org.state or ''}, {org.country or ''}".strip(', '))
            st.write(f"**Status:** {org.status or 'N/A'}")
            if org.description:
                st.caption(org.description)
                
        with col_b2:
            if org.investor_profile:
                st.write(f"**Investor Type:** {org.investor_profile.investor_type or 'N/A'}")
                st.write(f"**Preferred Stage:** {org.investor_profile.preferred_stage or 'N/A'}")
                st.write(f"**Founded Year:** {org.investor_profile.founded_year or 'N/A'}")
                if org.investor_profile.themes:
                    st.write(f"**Themes:** {', '.join(org.investor_profile.themes)}")
                if org.investor_profile.user_notes:
                    st.write(f"**User Notes:** {org.investor_profile.user_notes}")
            else:
                st.info("No extended investor profile available.")
            
        if org.program_memberships:
            st.divider()
            st.markdown("#### Program Memberships")
            for m in org.program_memberships:
                if m.cohort and m.cohort.program:
                    st.write(f"- 🎯 **{m.cohort.program.program_name}** — {m.cohort.cohort_name}")
                elif m.cohort:
                    st.write(f"- 🎯 {m.cohort.cohort_name}")
                else:
                    st.write("- 🎯 (Unlinked Membership)")
                    
        if org.fund_profiles:
            st.divider()
            st.markdown("#### Fund Profiles")
            
            fund_data = []
            for fund in org.fund_profiles:
                type_display = fund.fund_type or fund.investment_fund_type or "N/A"
                fund_data.append({
                    "Name": fund.fund_name,
                    "Vintage": fund.vintage_year or "",
                    "Raised": format_currency(fund.fund_size_raised),
                    "Target": format_currency(fund.fund_size_target),
                    "Type": type_display,
                    "Reputation": fund.market_reputation or "",
                    "Themes": ", ".join(fund.themes) if fund.themes else "",
                    "Imported": fund.created_at.strftime("%Y-%m-%d") if fund.created_at else ""
                })
            st.dataframe(fund_data, use_container_width=True, hide_index=True)
            
            with st.expander("Edit Funds & View Notes", expanded=False):
                for i, fund in enumerate(org.fund_profiles):
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.markdown(f"**{fund.fund_name}**")
                        if fund.user_notes:
                            st.write(f"**User Notes:** {fund.user_notes}")
                        if fund.description:
                            st.caption(fund.description)
                    with cols[1]:
                        if st.button("✏️ Edit", key=f"edit_fund_{fund.id}", use_container_width=True):
                            edit_fund_dialog(fund)
                    if i < len(org.fund_profiles) - 1:
                        st.divider()

        if org.program_profiles:
            st.divider()
            st.markdown("#### Programs & Cohorts")
            for prog in org.program_profiles:
                with st.expander(f"🎯 {prog.program_name} ({prog.program_type or 'Program'})"):
                    st.write(f"**Status:** {prog.status or 'N/A'}")
                    if prog.description:
                        st.write(prog.description)
                    
                    if prog.cohorts:
                        st.write("**Cohorts:**")
                        for cohort in prog.cohorts:
                            date_str = ""
                            if cohort.start_date:
                                date_str = f" ({cohort.start_date.strftime('%b %Y')} - {cohort.end_date.strftime('%b %Y') if cohort.end_date else 'Present'})"
                            st.write(f"- 📦 **{cohort.cohort_name}**{date_str}")
                            if cohort.description:
                                st.caption(f"  {cohort.description}")
                    else:
                        st.caption("No cohorts defined.")

        # People & Roles
        st.divider()
        st.subheader("👥 People & Roles")
        if org.roles:
            role_data = []
            for role in org.roles:
                p = role.person
                if p:
                    name = p.full_name or f"{p.first_name} {p.last_name}"
                    emails = ", ".join([e.email for e in p.emails]) if p.emails else ""
                    location = f"{p.city or ''}, {p.state or ''}".strip(", ")
                    if location == ",": location = ""
                    
                    years = ""
                    if role.start_date:
                        start_str = role.start_date.strftime("%Y")
                        end_str = role.end_date.strftime("%Y") if role.end_date else "Present"
                        years = f"{start_str} - {end_str}"
                    elif role.end_date:
                        years = f"Until {role.end_date.strftime('%Y')}"
                        
                    role_data.append({
                        "Name": name,
                        "Title": role.title or "",
                        "Years": years,
                        "Seniority": role.seniority_level or "",
                        "Location": location,
                        "Email": emails,
                        "LinkedIn": p.linkedin_url or ""
                    })
            if role_data:
                st.dataframe(role_data, use_container_width=True, hide_index=True)
            else:
                st.info("No people linked to this investor.")
        else:
            st.info("No people linked to this investor.")

        # Linked Source Documents
        st.divider()
        st.subheader("📄 Linked Source Documents")
        
        fund_ids = [str(f.id) for f in org.fund_profiles] if org.fund_profiles else []
        filters = [
            (EntityMatch.canonical_entity_type == "Organization") & (EntityMatch.canonical_entity_id == str(org.id))
        ]
        if fund_ids:
            filters.append((EntityMatch.canonical_entity_type == "FundProfile") & (EntityMatch.canonical_entity_id.in_(fund_ids)))
            
        doc_ids_subquery = db.query(SourceDocument.id).join(DocumentText).join(ExtractionJob).join(ExtractedEntity).join(EntityMatch).filter(
            or_(*filters)
        ).subquery()
        
        docs = db.query(SourceDocument).filter(SourceDocument.id.in_(doc_ids_subquery)).all()
        
        if docs:
            from market_comps.config import get_supabase_url
            
            for doc in docs:
                signed_url = get_supabase_url(doc.file_path) if doc.file_path else ""
                tz_time = format_est_datetime(doc.created_at)
                
                doc_name = "SEC Form D" if doc.document_type == "SEC_XML" else doc.document_type
                
                if doc.document_type == "SEC_XML":
                    url_display = f"[{doc_name}]({doc.source_url})"
                elif signed_url:
                    url_display = f"{doc.source_url} [(View)]({signed_url})"
                else:
                    url_display = f"[{doc.source_url}]({doc.source_url})" if str(doc.source_url).startswith("http") else doc.source_url
                
                st.markdown(f"- **{doc_name}**: {url_display} (Processed: {tz_time})")
        else:
            st.info("No documents linked to this investor.")

        # Raw Extracted Data
        st.divider()
        st.subheader("🧠 Raw Extracted Data")
        
        fund_ids = [str(f.id) for f in org.fund_profiles] if org.fund_profiles else []
        filters = [
            (EntityMatch.canonical_entity_type == "Organization") & (EntityMatch.canonical_entity_id == str(org.id))
        ]
        if fund_ids:
            filters.append((EntityMatch.canonical_entity_type == "FundProfile") & (EntityMatch.canonical_entity_id.in_(fund_ids)))
            
        matches = db.query(EntityMatch).filter(or_(*filters)).all()
        
        extracted_entities = list({m.extracted_entity.id: m.extracted_entity for m in matches if m.extracted_entity}.values())
        if extracted_entities:
            for ent in extracted_entities:
                job = ent.extraction_job
                schema_label = job.schema_name if job else "Unknown Schema"
                with st.expander(f"{schema_label} Data (Entity ID: {ent.id})"):
                    st.json(ent.extracted_payload_json)
        else:
            st.info("No raw extracted data found.")

        # Audit Trail
        st.divider()
        st.subheader("📜 Audit Trail")
        filters = [
            (AuditTrail.canonical_entity_type == "ORGANIZATION") & (AuditTrail.canonical_entity_id == str(org.id))
        ]
        if org.investor_profile:
            filters.append((AuditTrail.canonical_entity_type == "INVESTOR_PROFILE") & (AuditTrail.canonical_entity_id == str(org.investor_profile.id)))
            
        if fund_ids:
            filters.append((AuditTrail.canonical_entity_type == "FUND_PROFILE") & (AuditTrail.canonical_entity_id.in_(fund_ids)))
            
        role_ids = [str(role.id) for role in org.roles] if org.roles else []
        if role_ids:
            filters.append((AuditTrail.canonical_entity_type == "PERSON_ROLE") & (AuditTrail.canonical_entity_id.in_(role_ids)))
            
        audit = db.query(AuditTrail).options(
            joinedload(AuditTrail.extraction_job).joinedload(ExtractionJob.pipeline_run).joinedload(PipelineRun.source_documents)
        ).filter(
            or_(*filters)
        ).order_by(AuditTrail.created_at.desc()).limit(20).all()
        
        if audit:
            audit_data = []
            for a in audit:
                source_str = a.source or ""
                if source_str == "PIPELINE" and a.extraction_job and a.extraction_job.pipeline_run:
                    docs = a.extraction_job.pipeline_run.source_documents
                    if docs and docs[0].document_date:
                        source_str += f" (Doc: {docs[0].document_date})"
                        
                action_str = "Update" if a.mutation_type == "UPDATE" else "Create" if a.mutation_type == "CREATE" else a.mutation_type
                        
                audit_data.append({
                    "Date": format_est_datetime(a.created_at),
                    "Action": action_str,
                    "Field": a.field_name or "",
                    "Old Value": a.old_value or "",
                    "New Value": a.new_value or "",
                    "Source": source_str,
                    "User": a.created_by or "SYSTEM"
                })
            st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)
        else:
            st.info("No mutation entries found.")

with tab_firms:
    search_query = st.text_input("Search Investors...", placeholder="Search by name, domain, or website...")

    q = db.query(Organization).options(
        joinedload(Organization.investor_profile),
        joinedload(Organization.fund_profiles)
    ).filter(Organization.organization_type == "INVESTOR")

    if search_query:
        search_filter = f"%{search_query}%"
        q = q.filter(
            or_(
                Organization.name.ilike(search_filter),
                Organization.normalized_name.ilike(search_filter),
                Organization.primary_domain.ilike(search_filter)
            )
        )

    orgs = q.order_by(Organization.created_at.desc()).limit(200).all()

    # Prepare Dataframe
    data = []
    for o in orgs:
        row = {
            "ID": o.id,
            "Name": o.name,
            "Domain": o.primary_domain,
            "Website": o.website_url,
            "City": o.city,
            "State": o.state,
            "Status": o.status,
            "Created": o.created_at.strftime("%Y-%m-%d") if o.created_at else ""
        }
        if o.investor_profile:
            row["Inv Type"] = o.investor_profile.investor_type
            row["Pref Stage"] = o.investor_profile.preferred_stage
        
        fund_types = list(set([f.fund_type for f in o.fund_profiles if f.fund_type] + [f.investment_fund_type for f in o.fund_profiles if f.investment_fund_type]))
        row["Fund Types"] = ", ".join(fund_types)
        
        data.append(row)

    df = pd.DataFrame(data)

    if df.empty:
        st.info("No investor records found.")
    else:
        st.write("👆 *Select an investor row below to inspect full details.*")
        
        event = st.dataframe(
            df,
            key="grid_investors",
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True,
            use_container_width=True,
            column_config={
                "ID": None, # Hide ID
                "Website": st.column_config.LinkColumn("Website"),
                "Created": st.column_config.DateColumn("Created")
            }
        )
        
        selection = event.get("selection", {})
        rows = selection.get("rows", [])
        
        if rows:
            st.divider()
            selected_row_idx = rows[0]
            selected_investor_id = df.iloc[selected_row_idx]["ID"]
            display_investor_details(selected_investor_id)

with tab_funds:
    st.subheader("💰 All Fund Profiles")
    
    fund_q = db.query(FundProfile).join(Organization).order_by(FundProfile.created_at.desc())
    funds = fund_q.all()
    
    if not funds:
        st.info("No funds found in the CRM.")
    else:
        fund_data = []
        for f in funds:
            fund_data.append({
                "Fund ID": f.id,
                "Investment Firm": f.parent_organization.name if f.parent_organization else "Unknown",
                "Fund Name": f.fund_name,
                "Accession Number": f.accession_number or "",
                "Fund Type": f.fund_type or f.investment_fund_type or "",
                "Vintage": str(f.vintage_year) if f.vintage_year else "",
                "Raised": f.fund_size_raised or "",
                "Target": f.fund_size_target or "",
                "Status": f.status or "",
                "Reputation": f.market_reputation or "",
                "Description": f.description or ""
            })
            
        import pandas as pd
        fund_df = pd.DataFrame(fund_data)
        
        st.write("👆 *Select a fund row below to view full details in the main tab, or edit.*")
        event = st.dataframe(fund_df, use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun", key="grid_funds")
        
        selection = event.get("selection", {})
        rows = selection.get("rows", [])
        
        if rows:
            selected_row_idx = rows[0]
            selected_fund_id = fund_df.iloc[selected_row_idx]["Fund ID"]
            fund = db.query(FundProfile).filter(FundProfile.id == int(selected_fund_id)).first()
            if fund:
                edit_fund_dialog(fund)

with tab_add_firm:
    with st.form("investor_form", clear_on_submit=True):
        st.subheader("Organization Details")
        col1, col2 = st.columns(2)
        name = col1.text_input("Firm Name *")
        domain = col2.text_input("Primary Domain (Unique) *")
        city = col1.text_input("City")
        desc = st.text_area("Description")
        
        st.subheader("Investment Firm Profile")
        col3, col4 = st.columns(2)
        inv_type = col3.selectbox("Investment Firm Type", ["VC", "PE", "Angel", "CVC", "Family Office"])
        pref_stage = col4.text_input("Preferred Stage (e.g. Seed, Series A)")
        founded = col3.number_input("Founded Year", min_value=1800, max_value=2100, value=None, step=1)
        user_notes = st.text_area("User Notes")
        
        all_themes = get_all_investor_themes(db)
        selected_themes = st.multiselect("Themes", options=all_themes)
        new_themes = st.text_input("Add New Themes (comma separated)")
        
        submitted = st.form_submit_button("Create Investment Firm")
        if submitted:
            if not name or not domain:
                st.error("Name and Domain are required.")
            else:
                try:
                    org = db.query(Organization).filter_by(primary_domain=domain).first()
                    action_str = "updated" if org else "created"
                    
                    if org:
                        org.name = name
                        org.normalized_name = name.lower()
                        org.city = city
                        if desc: org.description = desc
                        org.organization_type = "INVESTOR"
                    else:
                        org = Organization(name=name, normalized_name=name.lower(), primary_domain=domain, city=city, description=desc, organization_type="INVESTOR")
                        db.add(org)
                        
                    db.flush()
                    
                    profile = db.query(InvestorProfile).filter_by(organization_id=org.id).first()
                    final_themes = list(selected_themes)
                    if new_themes:
                        final_themes.extend([t.strip() for t in new_themes.split(",") if t.strip()])
                    final_themes = list(set(final_themes))
                    
                    if profile:
                        if inv_type: profile.investor_type = inv_type
                        if pref_stage: profile.preferred_stage = pref_stage
                        if founded: profile.founded_year = founded
                        if user_notes: profile.user_notes = user_notes
                        if final_themes: profile.themes = final_themes
                    else:
                        profile = InvestorProfile(organization_id=org.id, investor_type=inv_type, preferred_stage=pref_stage, founded_year=founded, user_notes=user_notes, themes=final_themes if final_themes else None)
                        db.add(profile)
                        
                    db.commit()
                    st.success(f"Successfully {action_str} investor: {name}!")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving to DB: {str(e)}")

with tab_add_fund:
    orgs = db.query(Organization).filter_by(organization_type="INVESTOR").order_by(Organization.name).all()
    org_options = {org.id: f"{org.name} ({org.primary_domain})" for org in orgs}
    
    if not org_options:
        st.warning("You must create an Investment Firm Organization first before creating a Fund.")
    else:
        with st.form("fund_form", clear_on_submit=True):
            parent_id = st.selectbox("Parent Organization *", options=list(org_options.keys()), format_func=lambda x: org_options[x])
            
            col1, col2 = st.columns(2)
            name = col1.text_input("Fund Name *")
            f_type = col2.text_input("Fund Type (e.g. Flagship, Opportunity)")
            vintage = col1.number_input("Vintage Year", min_value=1980, max_value=2100, value=2024, step=1)
            size_raised = col1.text_input("Fund Size Raised (e.g. 500M)")
            size_target = col2.text_input("Fund Size Target (e.g. 750M)")
            
            reputation_opts = ["High", "Medium", "Low", "Emerging", ""]
            market_rep = col1.selectbox("Market Reputation", reputation_opts, index=4)
            
            desc = st.text_area("Description")
            user_notes = st.text_area("User Notes")
            
            all_themes = get_all_fund_themes(db)
            selected_themes = st.multiselect("Themes", options=all_themes)
            new_themes = st.text_input("Add New Themes (comma separated)")
            
            submitted = st.form_submit_button("Create Fund")
            if submitted:
                if not name:
                    st.error("Fund Name is required.")
                else:
                    try:
                        final_themes = list(selected_themes)
                        if new_themes:
                            final_themes.extend([t.strip() for t in new_themes.split(",") if t.strip()])
                        final_themes = list(set(final_themes))
                        
                        record = FundProfile(
                            parent_organization_id=parent_id,
                            fund_name=name,
                            fund_type=f_type,
                            vintage_year=vintage,
                            fund_size_raised=size_raised,
                            fund_size_target=size_target,
                            description=desc,
                            market_reputation=market_rep if market_rep else None,
                            user_notes=user_notes,
                            themes=final_themes if final_themes else None
                        )
                        db.add(record)
                        db.commit()
                        st.success(f"Successfully created Fund: {name}!")
                    except Exception as e:
                        db.rollback()
                        st.error(f"Error saving to DB: {str(e)}")
