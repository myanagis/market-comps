import streamlit as st
import json
from market_comps.db.session import get_db
import pandas as pd
from market_comps.db.models import (
    Pipeline, PipelineRun, SourceDocument, DocumentText, ExtractionJob, ExtractedEntity, ExtractedRelationship,
    Organization, ProgramProfile, ProgramCohort, FundProfile, EntityMatch
)
from market_comps.ingestion.pipeline_runner import run_pipeline

st.set_page_config(page_title="Data Ingestion", page_icon="📡", layout="wide")
st.title("📡 Data Ingestion Pipelines")

try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

PIPELINE_TYPES = [
    "PROGRAM_COMPANY_PAGE",
    "INVESTOR_PORTFOLIO_PAGE",
    "API_COMPANY_SEARCH",
    "CSV_IMPORT",
    "INVESTOR_PEOPLE_PAGE",
]

tab1, tab2, tab3, tab4 = st.tabs(["Pipelines", "Run Pipeline", "Pipeline Runs", "Extracted Data"])

# --- TAB 1: PIPELINES ---
with tab1:
    st.subheader("Create Pipeline")
    with st.form("new_pipeline_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        pipeline_name = col1.text_input("Pipeline Name *")
        pipeline_type = col2.selectbox("Pipeline Type", PIPELINE_TYPES)
        source_url = st.text_input("Source URL *")
        
        # Context links
        st.caption("Optional: Link this pipeline to an existing CRM record")
        col3, col4 = st.columns(2)
        
        orgs = db.query(Organization).order_by(Organization.name).all()
        org_opts = {0: "-- None --"}
        org_opts.update({o.id: o.name for o in orgs})
        org_id = col3.selectbox("Organization", options=list(org_opts.keys()), format_func=lambda x: org_opts[x])
        
        cohorts = db.query(ProgramCohort).all()
        cohort_opts = {0: "-- None --"}
        cohort_opts.update({ch.id: f"{ch.program.program_name} — {ch.cohort_name}" for ch in cohorts})
        cohort_id = col4.selectbox("Program Cohort", options=list(cohort_opts.keys()), format_func=lambda x: cohort_opts[x])
        
        # Config
        llm_instr = st.text_area("LLM Instruction (optional)", help="Custom instructions for the LLM extraction step")
        deep_scrape = st.checkbox("Enable Deep Scrape", help="Visit each company's profile page for rich data")
        
        if st.form_submit_button("Create Pipeline"):
            if not pipeline_name or not source_url:
                st.error("Pipeline Name and Source URL are required.")
            else:
                config = {}
                if llm_instr:
                    config["llm_instruction"] = llm_instr
                if deep_scrape:
                    config["deep_scrape"] = True
                    
                p = Pipeline(
                    pipeline_name=pipeline_name,
                    pipeline_type=pipeline_type,
                    source_url=source_url,
                    organization_id=org_id if org_id else None,
                    program_cohort_id=cohort_id if cohort_id else None,
                    schedule_type="MANUAL",
                    is_active=True,
                    config_json=config
                )
                db.add(p)
                db.commit()
                st.success(f"Created pipeline: {pipeline_name}")
                st.rerun()
    
    st.divider()
    st.subheader("Existing Pipelines")
    pipelines = db.query(Pipeline).order_by(Pipeline.created_at.desc()).all()
    if pipelines:
        for p in pipelines:
            cfg = p.config_json or {}
            deep_flag = "🔍" if cfg.get("deep_scrape") else ""
            cohort_label = f" → {p.program_cohort.program.program_name} / {p.program_cohort.cohort_name}" if p.program_cohort else ""
            
            with st.expander(f"{deep_flag} **{p.pipeline_name}** ({p.pipeline_type}){cohort_label}"):
                with st.form(f"edit_pipe_{p.id}"):
                    col1, col2 = st.columns(2)
                    new_name = col1.text_input("Name", value=p.pipeline_name, key=f"name_{p.id}")
                    new_type = col2.selectbox("Type", PIPELINE_TYPES, index=PIPELINE_TYPES.index(p.pipeline_type) if p.pipeline_type in PIPELINE_TYPES else 0, key=f"type_{p.id}")
                    new_url = st.text_input("Source URL", value=p.source_url or "", key=f"url_{p.id}")
                    
                    new_instr = st.text_area("LLM Instruction", value=cfg.get("llm_instruction", ""), key=f"instr_{p.id}")
                    new_deep = st.checkbox("Deep Scrape", value=cfg.get("deep_scrape", False), key=f"deep_{p.id}")
                    
                    # Cohort selector
                    cohort_opts_edit = {0: "-- None --"}
                    cohort_opts_edit.update({ch.id: f"{ch.program.program_name} — {ch.cohort_name}" for ch in cohorts})
                    current_cohort = p.program_cohort_id or 0
                    cohort_keys = list(cohort_opts_edit.keys())
                    cohort_idx = cohort_keys.index(current_cohort) if current_cohort in cohort_keys else 0
                    new_cohort = st.selectbox("Cohort", options=cohort_keys, format_func=lambda x: cohort_opts_edit[x], index=cohort_idx, key=f"cohort_{p.id}")
                    
                    if st.form_submit_button("💾 Save"):
                        p.pipeline_name = new_name
                        p.pipeline_type = new_type
                        p.source_url = new_url
                        p.program_cohort_id = new_cohort if new_cohort else None
                        new_cfg = dict(cfg)
                        new_cfg["llm_instruction"] = new_instr
                        new_cfg["deep_scrape"] = new_deep
                        p.config_json = new_cfg
                        db.commit()
                        st.success(f"Updated: {new_name}")
                        st.rerun()
    else:
        st.info("No pipelines yet.")


# --- TAB 2: RUN PIPELINE ---
with tab2:
    st.subheader("Run Pipeline")
    pipelines = db.query(Pipeline).filter_by(is_active=True).all()
    if not pipelines:
        st.warning("Create a pipeline first.")
    else:
        pipe_opts = {p.id: f"{p.pipeline_name} ({p.pipeline_type})" for p in pipelines}
        selected_id = st.selectbox("Select Pipeline", options=list(pipe_opts.keys()), format_func=lambda x: pipe_opts[x])
        
        if st.button("▶️ Run Pipeline", type="primary"):
            with st.spinner("Running pipeline..."):
                run = run_pipeline(db, selected_id)
                
                if run.run_status == "SUCCESS":
                    st.success(f"Pipeline completed! Processed {run.records_processed} entities, created {run.records_created} records.")
                    
                    # Show extracted entities
                    entities = db.query(ExtractedEntity).join(ExtractionJob).filter(ExtractionJob.pipeline_run_id == run.id).all()
                    if entities:
                        st.subheader("Extracted Entities")
                        for e in entities:
                            payload = e.extracted_payload_json or {}
                            icon = "🏢" if e.entity_type == "ORGANIZATION" else "👤"
                            with st.expander(f"{icon} {e.raw_name} ({e.entity_type})"):
                                st.json(payload)
                    
                    # Show logs
                    if run.logs_json:
                        with st.expander("📋 Pipeline Logs"):
                            st.json(run.logs_json)
                else:
                    st.error(f"Pipeline failed: {run.error_message}")


# --- TAB 3: PIPELINE RUNS ---
with tab3:
    st.subheader("Recent Pipeline Runs")
    runs = db.query(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(20).all()
    if runs:
        run_data = []
        for r in runs:
            run_data.append({
                "Run ID": r.id,
                "Pipeline": r.pipeline.pipeline_name if r.pipeline else "?",
                "Type": r.pipeline.pipeline_type if r.pipeline else "?",
                "Status": r.run_status,
                "Started": r.started_at,
                "Completed": r.completed_at,
                "Processed": r.records_processed,
                "Created": r.records_created,
                "Error": r.error_message or ""
            })
        st.dataframe(pd.DataFrame(run_data), use_container_width=True, hide_index=True)
        
        # Drill into a specific run
        run_ids = [r.id for r in runs]
        selected_run_id = st.selectbox("View run details", run_ids)
        if selected_run_id:
            selected_run = db.query(PipelineRun).filter_by(id=selected_run_id).first()
            if selected_run and selected_run.logs_json:
                with st.expander("📋 Run Logs", expanded=True):
                    st.json(selected_run.logs_json)
                    
            # Show source content
            raw_data = db.query(DocumentText).join(SourceDocument).filter(SourceDocument.pipeline_run_id == selected_run_id).all()
            if raw_data:
                with st.expander(f"📄 Source Content ({len(raw_data)} pages)"):
                    for rd in raw_data:
                        url_display = rd.source_document.source_url if rd.source_document else "?"
                        st.caption(f"**{rd.data_type}** — {url_display}")
                        st.code((rd.raw_content or "")[:3000], language=None)
    else:
        st.info("No runs yet.")


# --- TAB 4: EXTRACTED DATA ---
with tab4:
    st.subheader("Recent Extracted Entities")
    entities = db.query(ExtractedEntity).order_by(ExtractedEntity.created_at.desc()).limit(50).all()
    if entities:
        entity_data = []
        for e in entities:
            payload = e.extracted_payload_json or {}
            industry_val = payload.get("industry", "")
            if isinstance(industry_val, list):
                industry_val = ", ".join(industry_val)
                
            org_match = next((m.canonical_entity_id for m in e.matches if m.canonical_entity_type == "Organization"), "")
            person_match = next((m.canonical_entity_id for m in e.matches if m.canonical_entity_type == "Person"), "")
            
            entity_data.append({
                "Entity ID": e.id,
                "Job ID": e.extraction_job_id,
                "Type": e.entity_type,
                "Name": e.raw_name,
                "Industry": str(industry_val),
                "URL": str(payload.get("url", "")),
                "LinkedIn": str(payload.get("linkedin_url", "")),
                "Matched Org ID": org_match,
                "Matched Person ID": person_match,
            })
        st.dataframe(pd.DataFrame(entity_data), use_container_width=True, hide_index=True)
    else:
        st.info("No extracted entities yet.")
    
    st.divider()
    st.subheader("Recent Extracted Relationships")
    rels = db.query(ExtractedRelationship).order_by(ExtractedRelationship.created_at.desc()).limit(50).all()
    if rels:
        rel_data = []
        for r in rels:
            src_name = r.source_extracted_entity.raw_name if r.source_extracted_entity else "?"
            tgt_name = r.target_extracted_entity.raw_name if r.target_extracted_entity else "?"
            rel_data.append({
                "Rel ID": r.id,
                "Job ID": r.extraction_job_id,
                "Type": r.relationship_type,
                "Source": src_name,
                "Target": tgt_name,
                "Payload": json.dumps(r.relationship_payload_json) if r.relationship_payload_json else "",
            })
        st.dataframe(pd.DataFrame(rel_data), use_container_width=True, hide_index=True)
    else:
        st.info("No extracted relationships yet.")
