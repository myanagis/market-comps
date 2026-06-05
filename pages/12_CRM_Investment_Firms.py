import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import or_
from market_comps.db.session import get_db
from market_comps.db.models import (
    Organization, Person, InvestorProfile, FundProfile,
    ProgramMembership, PersonOrganizationRole, AuditTrail,
    ProgramProfile, ProgramCohort, EntityMatch, ExtractedEntity, ExtractionJob, PipelineRun, DocumentText, SourceDocument
)
from market_comps.ingestion.reconciler import log_mutation

st.set_page_config(page_title="Investment Firms Directory", page_icon="🏦", layout="wide")

st.title("🏦 Investment Firms Directory")
st.markdown("Browse and search investment firms and their funds in the CRM.")

tab_firms, tab_funds = st.tabs(["Investment Firms", "Fund Profiles"])

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

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
        with col4:
            preferred_stage = st.text_input("Preferred Stage", value=prof.preferred_stage if prof else "")
            
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
        
        st.markdown("#### Basic Information")
        st.write(f"**Domain:** {org.primary_domain} | **Website:** {org.website_url}")
        st.write(f"**LinkedIn:** {org.linkedin_url}")
        st.write(f"**Location:** {org.city}, {org.state}, {org.country}")
        st.write(f"**Status:** {org.status}")
        if org.description:
            st.info(org.description)
            
        if org.investor_profile:
            st.divider()
            st.markdown("#### Investor Profile")
            st.write(f"**Investor Type:** {org.investor_profile.investor_type}")
            st.write(f"**Preferred Stage:** {org.investor_profile.preferred_stage}")
            
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
            for fund in org.fund_profiles:
                with st.expander(f"💰 {fund.fund_name}"):
                    st.write(f"**Vintage:** {fund.vintage_year} | **Raised:** {fund.fund_size_raised} | **Target:** {fund.fund_size_target}")
                    st.write(f"**Type:** {fund.fund_type} | **Status:** {fund.status}")
                    if fund.description:
                        st.caption(fund.description)

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
        
        doc_ids_subquery = db.query(SourceDocument.id).join(DocumentText).join(ExtractionJob).join(ExtractedEntity).join(EntityMatch).filter(
            EntityMatch.canonical_entity_type == "Organization",
            EntityMatch.canonical_entity_id == str(org.id)
        ).subquery()
        
        docs = db.query(SourceDocument).filter(SourceDocument.id.in_(doc_ids_subquery)).all()
        
        if docs:
            from market_comps.config import get_supabase_url
            import zoneinfo
            eastern = zoneinfo.ZoneInfo("America/New_York")
            
            for doc in docs:
                signed_url = get_supabase_url(doc.file_path) if doc.file_path else ""
                if signed_url:
                    url_display = f"{doc.source_url} [(View)]({signed_url})"
                else:
                    url_display = f"[{doc.source_url}]({doc.source_url})" if str(doc.source_url).startswith("http") else doc.source_url
                
                tz_time = doc.created_at.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(eastern).strftime('%Y-%m-%d %I:%M %p ET') if doc.created_at else "Unknown Time"
                st.markdown(f"- **{doc.document_type}**: {url_display} (Processed: {tz_time})")
        else:
            st.info("No documents linked to this investor.")

        # Raw Extracted Data
        st.divider()
        st.subheader("🧠 Raw Extracted Data")
        
        matches = db.query(EntityMatch).filter_by(
            canonical_entity_type="Organization",
            canonical_entity_id=str(org.id)
        ).all()
        
        extracted_entities = [m.extracted_entity for m in matches if m.extracted_entity]
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
                        
                audit_data.append({
                    "Date": a.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Action": a.mutation_type,
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
        joinedload(Organization.investor_profile)
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
                "Street 1": f.street1 or "",
                "Street 2": f.street2 or "",
                "City": f.city or "",
                "State": f.state or "",
                "Country": f.country or "",
                "Zip Code": f.zip_code or "",
                "Description": f.description or ""
            })
            
        import pandas as pd
        fund_df = pd.DataFrame(fund_data)
        st.dataframe(fund_df, use_container_width=True, hide_index=True)
