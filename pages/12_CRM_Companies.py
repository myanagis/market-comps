import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import or_
from market_comps.db.session import get_db
from market_comps.db.models import (
    Organization, Person, CompanyProfile, FundProfile,
    ProgramMembership, PersonOrganizationRole, AuditTrail,
    ProgramProfile, ProgramCohort, EntityMatch, ExtractedEntity, ExtractionJob, PipelineRun, DocumentText, SourceDocument
)
from market_comps.ingestion.reconciler import log_mutation

st.set_page_config(page_title="Companies Directory", page_icon="🏢", layout="wide")
st.title("🏢 Companies Directory")

tab_dir, tab_add = st.tabs(["Companies Directory", "Add Company"])

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

def get_all_company_themes(db):
    profiles = db.query(CompanyProfile.themes).filter(CompanyProfile.themes.is_not(None)).all()
    themes = set()
    for (t,) in profiles:
        if t and isinstance(t, list):
            themes.update(t)
    return sorted(list(themes))

@st.dialog("Edit Company")
def edit_company_dialog(org):
    with st.form("edit_company"):
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
        st.write("Company Profile")
        col3, col4 = st.columns(2)
        prof = org.company_profile
        with col3:
            industry = st.text_input("Industry", value=prof.industry if prof else "")
            subind = st.text_input("Sub-Industry", value=prof.subindustry if prof else "")
        with col4:
            stage = st.text_input("Company Stage", value=prof.company_stage if prof else "")
            founded = st.number_input("Founded Year", value=prof.founded_year if prof and prof.founded_year else None, step=1, placeholder="YYYY")
            
        all_themes = get_all_company_themes(db)
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
                prof = CompanyProfile(organization_id=org.id)
                db.add(prof)
                db.flush()
                log_mutation(
                    db, "COMPANY_PROFILE", prof.id, "CREATE",
                    source="USER_EDIT", created_by=user
                )
                org.company_profile = prof
                
            check_and_update("COMPANY_PROFILE", prof.id, "industry", prof.industry, industry, prof)
            check_and_update("COMPANY_PROFILE", prof.id, "subindustry", prof.subindustry, subind, prof)
            check_and_update("COMPANY_PROFILE", prof.id, "company_stage", prof.company_stage, stage, prof)
            check_and_update("COMPANY_PROFILE", prof.id, "founded_year", prof.founded_year, founded if founded else None, prof)
            
            final_themes = list(selected_themes)
            if new_themes:
                final_themes.extend([t.strip() for t in new_themes.split(",") if t.strip()])
            final_themes = list(set(final_themes))
            
            check_and_update("COMPANY_PROFILE", prof.id, "themes", prof.themes, final_themes, prof)
            
            db.commit()
            st.rerun()

# Helper to render company details below the table
def display_company_details(company_id):
    org = db.query(Organization).options(
        joinedload(Organization.company_profile),
        joinedload(Organization.fund_profiles),
        joinedload(Organization.program_profiles).joinedload(ProgramProfile.cohorts),
        joinedload(Organization.program_memberships).joinedload(ProgramMembership.cohort).joinedload(ProgramCohort.program),
        joinedload(Organization.roles).joinedload(PersonOrganizationRole.person)
    ).filter(Organization.id == int(company_id)).first()

    if not org:
        st.error(f"Company with ID {company_id} not found.")
        return

    with st.container(border=True):
        col_h1, col_h2 = st.columns([5, 1])
        with col_h1:
            st.subheader(f"🏢 {org.name}")
        with col_h2:
            if st.button("✏️ Edit", key=f"edit_org_{org.id}", use_container_width=True):
                edit_company_dialog(org)
        
        st.markdown("#### Basic Information")
        st.write(f"**Domain:** {org.primary_domain} | **Website:** {org.website_url}")
        st.write(f"**LinkedIn:** {org.linkedin_url}")
        st.write(f"**Location:** {org.city}, {org.state}, {org.country}")
        st.write(f"**Status:** {org.status}")
        if org.description:
            st.info(org.description)
            
        if org.company_profile:
            st.divider()
            st.markdown("#### Company Profile")
            st.write(f"**Industry:** {org.company_profile.industry} | **Stage:** {org.company_profile.company_stage}")
            st.write(f"**Sub-Industry:** {org.company_profile.subindustry} | **Founded:** {org.company_profile.founded_year}")
            if org.company_profile.themes:
                st.write(f"**Themes:** {', '.join(org.company_profile.themes)}")
            
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
                st.info("No people linked to this company.")
        else:
            st.info("No people linked to this company.")

        # AI Profile Augmentation
        st.divider()
        col_aug1, col_aug2 = st.columns([3, 1])
        with col_aug1:
            st.subheader("🌐 Web Profile Augmentation")
        with col_aug2:
            if st.button("Augment via Web Search", key=f"btn_aug_{org.id}", use_container_width=True):
                with st.spinner("Searching the web and extracting evidence..."):
                    from market_comps.ingestion.company_augmentation import run_augmentation_pipeline
                    run_augmentation_pipeline(org.id)
                st.success("Augmentation Complete! Please refresh if the page does not reload automatically.")
                time.sleep(1)
                st.rerun()
                
        from market_comps.db.models import CompanyAugmentationReport
        latest_report = db.query(CompanyAugmentationReport).filter_by(organization_id=org.id).order_by(CompanyAugmentationReport.created_at.desc()).first()
        
        if latest_report and latest_report.status == "SUCCESS" and latest_report.extracted_data_json:
            import zoneinfo
            eastern = zoneinfo.ZoneInfo("America/New_York")
            tz_time = latest_report.created_at.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(eastern).strftime('%Y-%m-%d %I:%M %p ET') if latest_report.created_at else "Unknown"
            st.caption(f"*(Last augmented: {tz_time})*")
            
            exec_summary = latest_report.extracted_data_json.get("executive_summary")
            if exec_summary:
                st.markdown("##### Executive Summary")
                st.info(exec_summary)
            
            score_data = latest_report.scoring_json or {}
            if score_data:
                st.markdown("##### LLM Opinion Scores")
                score_cols = st.columns(len(score_data))
                for i, (section, s_data) in enumerate(score_data.items()):
                    with score_cols[i]:
                        score_val = s_data.get("score", 0)
                        conf = s_data.get("confidence", "Unknown")
                        color = "green" if score_val and score_val >= 8 else "orange" if score_val and score_val >= 5 else "red"
                        st.markdown(f"**{section.replace('_', ' ').title()}**")
                        st.markdown(f"### :{color}[{score_val}/10]")
                        st.caption(f"Conf: {conf}")
                        if s_data.get("reasoning"):
                            st.write(s_data.get("reasoning"))
                            
            st.markdown("##### Evidenced Data")
            for section, d in latest_report.extracted_data_json.items():
                if section == "executive_summary":
                    continue
                with st.expander(f"📦 {section.replace('_', ' ').title()}"):
                    summary_text = d.get("summary")
                    if summary_text:
                        st.info(summary_text)
                    
                    for quote in d.get("evidenced_data", []):
                        st.markdown(f"- \"{quote}\"")
        elif latest_report and latest_report.status == "FAILED":
            st.error(f"Last augmentation failed: {latest_report.error_message}")

        # Linked Source Documents
        st.divider()
        st.subheader("📄 Linked Source Documents")
        
        doc_ids_subquery_1 = db.query(SourceDocument.id).join(DocumentText).join(ExtractionJob).join(ExtractedEntity).join(EntityMatch).filter(
            EntityMatch.canonical_entity_type == "Organization",
            EntityMatch.canonical_entity_id == str(org.id)
        ).subquery()
        
        from market_comps.db.models import CompanyAugmentationReport, PipelineRun
        doc_ids_subquery_2 = db.query(SourceDocument.id).join(PipelineRun, SourceDocument.pipeline_run_id == PipelineRun.id).join(CompanyAugmentationReport, PipelineRun.id == CompanyAugmentationReport.pipeline_run_id).filter(
            CompanyAugmentationReport.organization_id == org.id
        ).subquery()
        
        docs = db.query(SourceDocument).filter(
            (SourceDocument.id.in_(doc_ids_subquery_1)) | (SourceDocument.id.in_(doc_ids_subquery_2))
        ).all()
        
        if docs:
            # Deduplicate by URL
            seen_urls = set()
            deduped_docs = []
            for doc in docs:
                url = str(doc.source_url).strip().lower()
                if url.endswith('/'): url = url[:-1]
                if url not in seen_urls:
                    seen_urls.add(url)
                    deduped_docs.append(doc)
            
            from market_comps.config import get_supabase_url
            import zoneinfo
            eastern = zoneinfo.ZoneInfo("America/New_York")
            
            for doc in deduped_docs:
                signed_url = get_supabase_url(doc.file_path) if doc.file_path else ""
                
                doc_label = doc.title if doc.title else doc.source_url
                
                if signed_url:
                    url_display = f"{doc_label} [(View)]({signed_url})"
                else:
                    url_display = f"[{doc_label}]({doc.source_url})" if str(doc.source_url).startswith("http") else doc_label
                
                tz_time = doc.created_at.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(eastern).strftime('%Y-%m-%d %I:%M %p ET') if doc.created_at else "Unknown Time"
                
                source_suffix = f" (via {doc.source_name})" if doc.source_name else ""
                st.markdown(f"- **{doc.document_type}**: {url_display}{source_suffix} (Processed: {tz_time})")
        else:
            st.info("No documents linked to this company.")

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
        if org.company_profile:
            filters.append((AuditTrail.canonical_entity_type == "COMPANY_PROFILE") & (AuditTrail.canonical_entity_id == str(org.company_profile.id)))
            
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

# Fetch and query data
with tab_dir:
    search_query = st.text_input("Search Companies...", placeholder="Search by name, domain, or website...")
    
    q = db.query(Organization).options(
        joinedload(Organization.company_profile)
    ).filter(Organization.organization_type == "COMPANY")

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
            "Status": o.status,
            "Created": o.created_at.strftime("%Y-%m-%d") if o.created_at else ""
        }
        if o.company_profile:
            row["Industry"] = o.company_profile.industry
            row["Stage"] = o.company_profile.company_stage
        data.append(row)

    df = pd.DataFrame(data)

    if df.empty:
        st.info("No company records found.")
    else:
        st.write("👆 *Select a company row below to inspect full details.*")
        
        event = st.dataframe(
            df,
            key="grid_companies",
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
            selected_company_id = df.iloc[selected_row_idx]["ID"]
            display_company_details(selected_company_id)

with tab_add:
    with st.form("company_form", clear_on_submit=True):
        st.subheader("Organization Details")
        col1, col2 = st.columns(2)
        name = col1.text_input("Company Name *")
        domain = col2.text_input("Primary Domain (Unique) *")
        city = col1.text_input("City")
        desc = st.text_area("Description")
        
        st.subheader("Company Profile")
        col3, col4 = st.columns(2)
        founded = col3.number_input("Founded Year", min_value=1800, max_value=2100, value=None, placeholder="YYYY")
        industry = col4.text_input("Industry")
        stage = col3.text_input("Company Stage")
        
        all_themes = get_all_company_themes(db)
        selected_themes = st.multiselect("Themes", options=all_themes)
        new_themes = st.text_input("Add New Themes (comma separated)")
        
        from market_comps.db.models import Investment
        st.subheader("Add Investment Firm (Optional)")
        investors = db.query(Organization).filter_by(organization_type="INVESTOR").order_by(Organization.name).all()
        investor_opts = {0: "-- None --"}
        investor_opts.update({i.id: i.name for i in investors})
        
        col5, col6 = st.columns(2)
        linked_investor_id = col5.selectbox("Select Investment Firm", options=list(investor_opts.keys()), format_func=lambda x: investor_opts[x])
        inv_round = col6.text_input("Round (e.g. Seed, Series A)")
        inv_amount = col5.text_input("Amount (e.g. $2M)")
        inv_date = col6.date_input("Investment Date", value=None)
        
        submitted = st.form_submit_button("Create Company")
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
                        org.organization_type = "COMPANY"
                    else:
                        org = Organization(name=name, normalized_name=name.lower(), primary_domain=domain, city=city, description=desc, organization_type="COMPANY")
                        db.add(org)
                        
                    db.flush() # get ID
                    
                    profile = db.query(CompanyProfile).filter_by(organization_id=org.id).first()
                    
                    final_themes = list(selected_themes)
                    if new_themes:
                        final_themes.extend([t.strip() for t in new_themes.split(",") if t.strip()])
                    final_themes = list(set(final_themes))
                    
                    if profile:
                        if founded: profile.founded_year = founded
                        if industry: profile.industry = industry
                        if stage: profile.company_stage = stage
                        if final_themes: profile.themes = final_themes
                    else:
                        profile = CompanyProfile(organization_id=org.id, founded_year=founded, industry=industry, company_stage=stage, themes=final_themes if final_themes else None)
                        db.add(profile)
                        
                    db.commit()
                    
                    if linked_investor_id != 0:
                        inv = Investment(
                            investor_organization_id=linked_investor_id,
                            company_organization_id=org.id,
                            round_type=inv_round,
                            amount=inv_amount,
                            investment_date=inv_date
                        )
                        db.add(inv)
                        db.commit()
                        st.success(f"Successfully added investment from {investor_opts[linked_investor_id]}!")
                        
                    st.success(f"Successfully {action_str} company: {name}!")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving to DB: {str(e)}")
