with open("pages/13_Data_Ingestion.py", "r", encoding="utf-8") as f:
    content = f.read()

import re

# We want to replace the drill-down section in tab3.
# Let's find:
#         # Drill into a specific run
#         run_ids = [r.id for r in runs]
#         selected_run_id = st.selectbox("View run details", run_ids)
#         if selected_run_id:
#             selected_run = db.query(PipelineRun).filter_by(id=selected_run_id).first()

new_drill_down_code = """        # Drill into a specific run
        run_ids = [r.id for r in runs]
        selected_run_id = st.selectbox("View run details", run_ids)
        if selected_run_id:
            from sqlalchemy.orm import joinedload
            selected_run = db.query(PipelineRun).options(
                joinedload(PipelineRun.steps),
                joinedload(PipelineRun.source_documents).joinedload(SourceDocument.texts),
                joinedload(PipelineRun.extraction_jobs).joinedload(ExtractionJob.entities)
            ).filter_by(id=selected_run_id).first()
            
            if selected_run:
                st.markdown(f"### Run Diagnostics: #{selected_run.id}")
                
                if selected_run.run_status == "FAILED" and selected_run.error_message:
                    st.error(f"**Pipeline Error:**\\n\\n```\\n{selected_run.error_message}\\n```")
                
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
                            for t in doc.texts:
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
                            "LLM Tokens": job.tokens_used,
                            "Entities Found": len(job.entities)
                        })
                    st.dataframe(pd.DataFrame(job_data), use_container_width=True, hide_index=True)
                    
                    with st.expander("🧠 View Raw Extracted Entities"):
                        for job in selected_run.extraction_jobs:
                            for ent in job.entities:
                                st.markdown(f"**Entity {ent.id} | {ent.entity_type} | {ent.raw_name}**")
                                st.json(ent.extracted_payload_json)
                else:
                    st.info("No extraction jobs found. (Note: SEC Form D uses hardcoded extraction, so it skips ExtractionJob tables!)")

                # 4. Global Run Logs
                if selected_run.logs_json:
                    with st.expander("📋 Global Run Logs JSON", expanded=True):
                        st.json(selected_run.logs_json)
"""

# Now we need to splice this into the file.
pattern = r"        # Drill into a specific run\s*run_ids = \[r\.id for r in runs\]\s*selected_run_id = st\.selectbox\(\"View run details\", run_ids\)\s*if selected_run_id:.*?(?=\n# --- TAB 4: EXTRACTED DATA ---|\Z)"

new_content = re.sub(pattern, new_drill_down_code, content, flags=re.DOTALL)

with open("pages/13_Data_Ingestion.py", "w", encoding="utf-8") as f:
    f.write(new_content)
print("done")
