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

from market_comps.ingestion.registry import FETCHERS, PREPARERS, EXTRACTORS, NORMALIZERS

CONNECTOR_TYPES = list(FETCHERS.keys())
PARSER_TYPES = list(PREPARERS.keys())
NORMALIZER_TYPES = list(NORMALIZERS.keys())

tab1, tab2, tab3, tab4 = st.tabs(["Pipelines", "Run Pipeline", "Pipeline Run History", "Extracted Data"])

# --- TAB 1: PIPELINES ---
with tab1:
    st.subheader("Create Pipeline")
    with st.form("new_pipeline_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        pipeline_name = col1.text_input("Pipeline Name *")
        connector_type = col2.selectbox("Connector (Fetch)", CONNECTOR_TYPES)
        
        col_p, col_n = st.columns(2)
        parser_type = col_p.selectbox("Parser (Extract)", PARSER_TYPES)
        normalizer_type = col_n.selectbox("Normalizer (Map)", NORMALIZER_TYPES)

        source_url = st.text_input("Source URL (Optional for SEC)")
        
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
        col_c1, col_c2, col_c3 = st.columns(3)
        deep_scrape = col_c1.checkbox("Enable Deep Scrape", help="Visit each company's profile page for rich data")
        days_back = col_c2.number_input("Days Back (SEC)", min_value=1, max_value=365, value=7, help="How many days of SEC Form D filings to fetch")
        max_filings = col_c3.number_input("Max Filings (SEC)", min_value=1, max_value=1000, value=50, help="Maximum number of filings to actually process per run (to limit cost)")
        
        if st.form_submit_button("Create Pipeline"):
            if not pipeline_name or not source_url:
                st.error("Pipeline Name and Source URL are required.")
            else:
                config = {}
                if llm_instr:
                    config["llm_instruction"] = llm_instr
                if deep_scrape:
                    config["deep_scrape"] = True
                if days_back != 7:
                    config["days_back"] = days_back
                if max_filings != 10:
                    config["max_filings_to_process"] = max_filings
                    
                p = Pipeline(
                    pipeline_name=pipeline_name,
                    connector_type=connector_type,
                    parser_type=parser_type,
                    normalizer_type=normalizer_type,
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
            
            with st.expander(f"{deep_flag} **{p.pipeline_name}** ({p.connector_type}){cohort_label}"):
                with st.form(f"edit_pipe_{p.id}"):
                    col1, col2 = st.columns(2)
                    new_name = col1.text_input("Name", value=p.pipeline_name, key=f"name_{p.id}")
                    new_conn = col2.selectbox("Connector", CONNECTOR_TYPES, index=CONNECTOR_TYPES.index(p.connector_type) if p.connector_type in CONNECTOR_TYPES else 0, key=f"conn_{p.id}")
                    
                    col_p, col_n = st.columns(2)
                    new_parser = col_p.selectbox("Parser", PARSER_TYPES, index=PARSER_TYPES.index(p.parser_type) if p.parser_type in PARSER_TYPES else 0, key=f"pars_{p.id}")
                    new_norm = col_n.selectbox("Normalizer", NORMALIZER_TYPES, index=NORMALIZER_TYPES.index(p.normalizer_type) if p.normalizer_type in NORMALIZER_TYPES else 0, key=f"norm_{p.id}")

                    new_url = st.text_input("Source URL", value=p.source_url or "", key=f"url_{p.id}")
                    
                    new_instr = st.text_area("LLM Instruction", value=cfg.get("llm_instruction", ""), key=f"instr_{p.id}")
                    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
                    new_deep = col_cfg1.checkbox("Deep Scrape", value=cfg.get("deep_scrape", False), key=f"deep_{p.id}")
                    new_days = col_cfg2.number_input("Days Back", min_value=1, max_value=365, value=cfg.get("days_back", 7), key=f"days_{p.id}")
                    new_max = col_cfg3.number_input("Max Filings", min_value=1, max_value=1000, value=cfg.get("max_filings_to_process", 10), key=f"max_{p.id}")
                    
                    # Cohort selector
                    cohort_opts_edit = {0: "-- None --"}
                    cohort_opts_edit.update({ch.id: f"{ch.program.program_name} — {ch.cohort_name}" for ch in cohorts})
                    current_cohort = p.program_cohort_id or 0
                    cohort_keys = list(cohort_opts_edit.keys())
                    cohort_idx = cohort_keys.index(current_cohort) if current_cohort in cohort_keys else 0
                    new_cohort = st.selectbox("Cohort", options=cohort_keys, format_func=lambda x: cohort_opts_edit[x], index=cohort_idx, key=f"cohort_{p.id}")
                    
                    if st.form_submit_button("💾 Save"):
                        p.pipeline_name = new_name
                        p.connector_type = new_conn
                        p.parser_type = new_parser
                        p.normalizer_type = new_norm
                        p.source_url = new_url
                        p.program_cohort_id = new_cohort if new_cohort else None
                        new_cfg = dict(cfg)
                        if new_instr: new_cfg["llm_instruction"] = new_instr
                        elif "llm_instruction" in new_cfg: del new_cfg["llm_instruction"]
                        new_cfg["deep_scrape"] = new_deep
                        new_cfg["days_back"] = new_days
                        new_cfg["max_filings_to_process"] = new_max
                        p.config_json = new_cfg
                        db.commit()
                        st.success(f"Updated: {new_name}")
                        st.rerun()
    else:
        st.info("No pipelines yet.")


# --- TAB 2: RUN PIPELINE ---
with tab2:
    st.subheader("Run Pipeline")
    pipelines = db.query(Pipeline).filter(Pipeline.is_active == True).all()
    if not pipelines:
        st.warning("Create a pipeline first.")
    else:
        pipe_opts = {p.id: f"{p.pipeline_name} ({p.connector_type})" for p in pipelines}
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


# --- TAB 3: PIPELINE RUN HISTORY ---
with tab3:
    st.subheader("Recent Ingestion Runs")
    runs = db.query(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(20).all()
    if runs:
        run_data = []
        for r in runs:
            run_data.append({
                "Run ID": r.id,
                "Source": r.pipeline.pipeline_name if r.pipeline else "Document Upload",
                "Type": r.pipeline.connector_type if r.pipeline else "DOCUMENT",
                "Status": r.run_status,
                "Started": r.started_at,
                "Completed": r.completed_at,
                "Processed": r.records_processed,
                "Created": r.records_created,
                "Updated": r.records_updated,
                "Error": r.error_message or ""
            })
        st.dataframe(pd.DataFrame(run_data), use_container_width=True, hide_index=True)
        
        # Drill into a specific run
        run_ids = [r.id for r in runs]
        selected_run_id = st.selectbox("View run details", run_ids)
        if selected_run_id:
            from sqlalchemy.orm import joinedload
            selected_run = db.query(PipelineRun).options(
                joinedload(PipelineRun.steps),
                joinedload(PipelineRun.source_documents).joinedload(SourceDocument.document_texts),
                joinedload(PipelineRun.extraction_jobs).joinedload(ExtractionJob.extracted_entities)
            ).filter_by(id=selected_run_id).first()
            
            if selected_run:
                st.markdown(f"### Run Diagnostics: #{selected_run.id}")
                
                if selected_run.run_status == "FAILED" and selected_run.error_message:
                    st.error(f"**Pipeline Error:**\n\n```\n{selected_run.error_message}\n```")
                
                # Show run config
                run_cfg = selected_run.pipeline.config_json if selected_run.pipeline else {}
                with st.expander("⚙️ Pipeline Configuration", expanded=False):
                    st.json(run_cfg)
                
                # 1. Show Run Steps
                if selected_run.steps:
                    st.markdown("#### 1. Pipeline Execution Steps (`PipelineRunStep`)")
                    step_data = []
                    for step in selected_run.steps:
                        step_data.append({
                            "Order": step.step_order,
                            "Phase": step.step_name,
                            "Method": step.method,
                            "Status": step.status,
                            "Started": step.started_at.strftime("%H:%M:%S") if step.started_at else "",
                            "Duration": f"{(step.completed_at - step.started_at).total_seconds():.1f}s" if step.completed_at and step.started_at else "",
                        })
                    st.dataframe(pd.DataFrame(step_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No pipeline steps recorded for this run.")

                # 2. Show Source Documents
                if selected_run.source_documents:
                    st.markdown("#### 2. Fetched Source Documents (`SourceDocument`)")
                    doc_data = []
                    for doc in selected_run.source_documents:
                        doc_data.append({
                            "Doc ID": doc.id,
                            "Type": doc.document_type,
                            "URL / Path": doc.source_url or doc.file_path,
                            "Created": doc.created_at.strftime("%Y-%m-%d %H:%M:%S") if doc.created_at else "",
                            "Hash": doc.content_hash[:10] + "..." if doc.content_hash else ""
                        })
                    st.dataframe(pd.DataFrame(doc_data), use_container_width=True, hide_index=True)
                    
                    with st.expander("📄 View Raw Document Texts"):
                        for doc in selected_run.source_documents:
                            for t in doc.document_texts:
                                st.caption(f"**Doc {doc.id} | Text {t.id} | {t.data_type}**")
                                st.code((t.raw_content or "")[:1500] + ("..." if len(t.raw_content or "") > 1500 else ""), language=None)
                else:
                    st.info("No source documents fetched.")

                # 3. Show Extraction Jobs & Entities
                if selected_run.extraction_jobs:
                    st.markdown("#### 3. Data Extraction Jobs (`ExtractionJob`)")
                    job_data = []
                    for job in selected_run.extraction_jobs:
                        job_data.append({
                            "Job ID": job.id,
                            "Schema": job.schema_name,
                            "Status": job.status,
                            "LLM Tokens": job.tokens_used if hasattr(job, "tokens_used") else 0,
                            "Entities Found": len(job.extracted_entities)
                        })
                    st.dataframe(pd.DataFrame(job_data), use_container_width=True, hide_index=True)
                    
                    with st.expander("🧠 View Raw Extracted Entities"):
                        for job in selected_run.extraction_jobs:
                            for ent in job.extracted_entities:
                                st.markdown(f"**Entity {ent.id} | {ent.entity_type} | {ent.raw_name}**")
                                st.json(ent.extracted_payload_json)
                else:
                    st.info("No extraction jobs found. (Note: SEC Form D uses hardcoded extraction, so it skips ExtractionJob tables!)")

                # 4. Global Run Logs
                if selected_run.logs_json:
                    with st.expander("📋 Global Run Logs JSON", expanded=True):
                        st.json(selected_run.logs_json)

# --- TAB 4: EXTRACTED DATA ---
with tab4:
    st.subheader("Recent Extracted Entities")
    entities = db.query(ExtractedEntity).order_by(ExtractedEntity.created_at.desc()).limit(50).all()
    if entities:
        for e in entities:
            payload = e.extracted_payload_json or {}
            
            org_match = next((m.canonical_entity_id for m in e.matches if m.canonical_entity_type == "Organization"), "")
            person_match = next((m.canonical_entity_id for m in e.matches if m.canonical_entity_type == "Person"), "")
            
            match_label = ""
            if org_match: match_label = f" → Org ID: {org_match}"
            elif person_match: match_label = f" → Person ID: {person_match}"
            
            with st.expander(f"[{e.entity_type}] {e.raw_name}{match_label} (Job {e.extraction_job_id})"):
                st.json(payload)
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
