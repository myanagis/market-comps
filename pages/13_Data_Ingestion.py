import streamlit as st
import json
from market_comps.db.session import get_db
import pandas as pd
from market_comps.db.models import DataSource, IngestionConfig, IngestionJob, RawEntity, EntityUpdate
from market_comps.ingestion.api_runner import run_ingestion_config

st.set_page_config(page_title="Data Ingestion", page_icon="📡", layout="wide")
st.title("📡 Data Ingestion & Scrape Jobs")

try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Data Sources", "Ingestion Configs", "Run Manual Job"])

# --- TAB 1: DATA SOURCES ---
with tab1:
    st.subheader("Add Data Source")
    with st.form("new_source_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        name = col1.text_input("Source Name (e.g. A16Z Speedrun, Crunchbase API) *")
        base_url = col2.text_input("Base URL (e.g. https://a16z.com)")
        desc = st.text_area("Description")
        
        submitted = st.form_submit_button("Create Data Source")
        if submitted:
            if not name:
                st.error("Name is required.")
            else:
                ds = DataSource(source_name=name, base_url=base_url, description=desc)
                db.add(ds)
                db.commit()
                st.success(f"Created Data Source: {name}")
                st.rerun()
                
    st.divider()
    st.subheader("Existing Data Sources")
    sources = db.query(DataSource).all()
    if sources:
        for ds in sources:
            with st.expander(f"**{ds.source_name}** ({ds.base_url})", expanded=False):
                st.write(f"ID: {ds.id}")
                st.write(f"Description: {ds.description}")
    else:
        st.info("No data sources configured yet.")

# --- TAB 2: INGESTION CONFIGS ---
with tab2:
    st.subheader("Add Ingestion Config")
    sources = db.query(DataSource).all()
    if not sources:
        st.warning("Create a Data Source first.")
    else:
        source_opts = {s.id: s.source_name for s in sources}
        with st.form("new_config_form", clear_on_submit=True):
            source_id = st.selectbox("Data Source", options=list(source_opts.keys()), format_func=lambda x: source_opts[x])
            
            col1, col2, col3 = st.columns([2, 1, 1])
            config_name = col1.text_input("Config Name (e.g. Fetch Speedrun Cohort 6) *")
            endpoint = col1.text_input("Endpoint URL / Path (e.g. /games-speedrun/)")
            ingestion_type = col2.selectbox("Type", ["SCRAPE", "API"])
            method = col3.selectbox("HTTP Method", ["GET", "POST"])
            
            llm_instr = st.text_area("LLM Instruction (For SCRAPE type only)", 
                help="e.g. 'Extract all companies from this page, and tag with the program Speedrun Cohort 6'")
            
            submitted = st.form_submit_button("Create Config")
            if submitted:
                if not config_name:
                    st.error("Config Name is required.")
                else:
                    meta = {}
                    if llm_instr:
                        meta["llm_instruction"] = llm_instr
                    
                    cfg = IngestionConfig(
                        data_source_id=source_id,
                        config_name=config_name,
                        ingestion_type=ingestion_type,
                        endpoint_url=endpoint,
                        http_method=method,
                        metadata_json=meta
                    )
                    db.add(cfg)
                    db.commit()
                    st.success(f"Created Config: {config_name}")
                    st.rerun()
                    
    st.divider()
    st.subheader("Existing Configs")
    configs = db.query(IngestionConfig).all()
    if configs:
        for c in configs:
            with st.expander(f"**{c.config_name}** ({c.ingestion_type})"):
                st.write(f"Data Source ID: {c.data_source_id}")
                st.write(f"Endpoint: {c.http_method} {c.endpoint_url}")
                st.json(c.metadata_json or {})
    else:
        st.info("No configs yet.")

# --- TAB 3: RUN MANUAL JOB ---
with tab3:
    st.subheader("Run Ingestion Job")
    configs = db.query(IngestionConfig).all()
    if not configs:
        st.warning("Create an Ingestion Config first.")
    else:
        cfg_opts = {c.id: f"{c.data_source.source_name} - {c.config_name} ({c.ingestion_type})" for c in configs}
        selected_cfg = st.selectbox("Select Config to Run", options=list(cfg_opts.keys()), format_func=lambda x: cfg_opts[x])
        
        if st.button("▶️ Run Job", type="primary"):
            with st.spinner("Executing API / Scrape Job..."):
                job = run_ingestion_config(db, selected_cfg, triggered_by="MANUAL")
                
                if job.job_status == "SUCCESS":
                    st.success(f"Job completed successfully! Extracted {job.records_processed} records.")
                    
                    st.subheader("Job Logs / Extracted Entities")
                    
                    if job.job_logs_json and "extracted_companies" in job.job_logs_json:
                        companies = job.job_logs_json["extracted_companies"]
                        for c in companies:
                            status = c.get("__reconciliation_status__", "UNKNOWN")
                            icon = "🆕" if status == "CREATED_ORG" else ("🔄" if status == "UPDATED_ORG" else "⏺️")
                            
                            with st.expander(f"{icon} **{c.get('name', 'Unknown')}** ({status})"):
                                st.write(f"**URL:** {c.get('url', '')} | **LinkedIn:** {c.get('linkedin_url', '')}")
                                st.write(f"**Industry:** {c.get('industry', 'N/A')} | **Founded:** {c.get('founded_year', 'N/A')}")
                                st.write(f"**Description:** {c.get('description', '')}")
                                
                                founders = c.get("founders", [])
                                if founders:
                                    st.write("**Founders:**")
                                    for f in founders:
                                        st.write(f"- {f.get('first_name', '')} {f.get('last_name', '')} ({f.get('title', 'Founder')}) — {f.get('email', '')}")
                                        
                        st.divider()
                        st.caption("Raw Logs:")
                        st.json(job.job_logs_json.get("llm_usage", {}))
                    else:
                        st.json(job.job_logs_json)
                    
                else:
                    st.error(f"Job failed: {job.error_message}")
                    
    st.divider()
    st.subheader("Recent Jobs")
    recent_jobs = db.query(IngestionJob).order_by(IngestionJob.started_at.desc()).limit(10).all()
    if recent_jobs:
        job_data = []
        for j in recent_jobs:
            job_data.append({
                "Job ID": j.id,
                "Config ID": j.ingestion_config_id,
                "Status": j.job_status,
                "Triggered By": j.triggered_by,
                "Started At": j.started_at.strftime('%Y-%m-%d %H:%M:%S') if j.started_at else None,
                "Completed At": j.completed_at.strftime('%Y-%m-%d %H:%M:%S') if j.completed_at else None,
                "Processed": j.records_processed,
                "Created": j.records_created,
                "Updated": j.records_updated,
                "Failed": j.records_failed,
                "Error": j.error_message
            })
        st.dataframe(pd.DataFrame(job_data), use_container_width=True, hide_index=True)
            
    st.divider()
    st.subheader("Recent Raw Entities (Reconciliation Log)")
    recent_entities = db.query(RawEntity).order_by(RawEntity.detected_at.desc()).limit(50).all()
    
    if recent_entities:
        data = []
        for e in recent_entities:
            # Check what actions the reconciliation engine took for this entity
            updates = db.query(EntityUpdate).filter(EntityUpdate.raw_entity_id == e.id).all()
            
            org_status = "N/A"
            person_statuses = []
            
            for u in updates:
                if u.organization_id:
                    org_status = f"{u.update_action} (Org {u.organization_id})"
                elif u.person_id:
                    person_statuses.append(f"{u.update_action} (Person {u.person_id})")
                    
            payload = e.raw_payload_json or {}
            founders = payload.get("founders", [])
            founder_names = [f"{f.get('first_name', '')} {f.get('last_name', '')}" for f in founders]
            
            data.append({
                "Job ID": e.ingestion_job_id,
                "Raw Name": e.raw_name,
                "Industry": payload.get("industry", "N/A"),
                "Founded": payload.get("founded_year", "N/A"),
                "Founders": ", ".join(founder_names) if founder_names else "N/A",
                "Description": payload.get("description", "N/A"),
                "Matched Org ID": e.matched_organization_id,
                "Organization Action": org_status,
                "Person Actions": ", ".join(person_statuses) if person_statuses else "N/A",
                "Detected At": e.detected_at.strftime('%Y-%m-%d %H:%M'),
                "Source URL": e.source_url
            })
            
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No raw entities extracted yet.")
