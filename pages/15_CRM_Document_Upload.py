import streamlit as st
import hashlib
from datetime import datetime
import pdfplumber
import io

from market_comps.db.session import get_db
from market_comps.db.models import (
    Pipeline, PipelineRun, SourceDocument, DocumentText, ExtractionJob, ExtractedEntity, ExtractedRelationship, Organization
)
from market_comps.ingestion.extractor import extract_entities_from_text
from market_comps.ingestion.reconciler import reconcile_all

st.set_page_config(page_title="Document Upload", page_icon="📄", layout="wide")
st.title("📄 Upload CRM Document")
st.write("Upload a document to automatically extract and link companies, people, and other entities to the CRM.")

try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

PIPELINE_TYPES = [
    "DOCUMENT_ENTITIES",
    "PROGRAM_COMPANY_PAGE",
    "INVESTOR_PORTFOLIO_PAGE",
]

with st.form("upload_document_form", clear_on_submit=False):
    uploaded_file = st.file_uploader("Choose a document", type=["pdf", "txt"])
    
    col1, col2 = st.columns(2)
    pipeline_type = col1.selectbox("Extraction Schema", PIPELINE_TYPES, index=0, help="What kind of entities should the AI look for?")
    
    # Optional linked Organization
    orgs = db.query(Organization).order_by(Organization.name).all()
    org_opts = {0: "-- None --"}
    org_opts.update({o.id: o.name for o in orgs})
    linked_org_id = col2.selectbox(
        "Linked Organization (Optional)", 
        options=list(org_opts.keys()), 
        format_func=lambda x: org_opts[x],
        help="Providing a linked organization helps the LLM understand the context of the document."
    )
    
    custom_instructions = st.text_area("Additional LLM Instructions (Optional)", help="E.g., 'Pay special attention to founders.'")
    
    submitted = st.form_submit_button("Upload & Process", type="primary")

if submitted:
    if not uploaded_file:
        st.error("Please upload a file.")
    else:
        with st.spinner("Processing document..."):
            # 1. Read file content
            file_name = uploaded_file.name
            file_bytes = uploaded_file.read()
            text_content = ""
            
            if file_name.lower().endswith(".pdf"):
                try:
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        for page in pdf.pages:
                            text_content += page.extract_text() + "\n"
                except Exception as e:
                    st.error(f"Failed to parse PDF: {e}")
                    st.stop()
            else:
                try:
                    text_content = file_bytes.decode("utf-8")
                except Exception as e:
                    st.error(f"Failed to parse text file: {e}")
                    st.stop()
            
            if not text_content.strip():
                st.error("Document appears to be empty or contains no extractable text.")
                st.stop()

            # 2. Build instructions context
            final_instructions = custom_instructions
            if linked_org_id != 0:
                linked_org = db.query(Organization).get(linked_org_id)
                if linked_org:
                    ctx_note = f"NOTE: This document is related to the organization '{linked_org.name}'. Pay special attention to their properties and relationships."
                    final_instructions = f"{ctx_note}\n{final_instructions}".strip()

            # 3. Create Pipeline & DB records
            content_hash = hashlib.sha256(text_content.encode()).hexdigest()
            
            pipeline = Pipeline(
                pipeline_name=f"Upload: {file_name}",
                pipeline_type=pipeline_type,
                source_url=file_name,
                organization_id=linked_org_id if linked_org_id != 0 else None,
                schedule_type="MANUAL",
                is_active=True,
                config_json={"llm_instruction": final_instructions}
            )
            db.add(pipeline)
            db.flush()
            
            run = PipelineRun(
                pipeline_id=pipeline.id,
                run_status="RUNNING",
                started_at=datetime.utcnow()
            )
            db.add(run)
            db.flush()
            
            source_doc = SourceDocument(
                pipeline_run_id=run.id,
                document_type="PDF" if file_name.lower().endswith(".pdf") else "TEXT",
                source_url=file_name,
                content_hash=content_hash,
            )
            db.add(source_doc)
            db.flush()
            
            doc_text = DocumentText(
                source_document_id=source_doc.id,
                data_type="DOCUMENT_TEXT",
                raw_content=text_content,
                content_hash=content_hash,
            )
            db.add(doc_text)
            db.flush()
            
            job = ExtractionJob(
                pipeline_run_id=run.id,
                document_text_id=doc_text.id,
                schema_name=pipeline_type,
                status="IN_PROGRESS",
                started_at=datetime.utcnow()
            )
            db.add(job)
            db.flush()
            
            # 4. Run LLM Extraction
            try:
                st.info("🤖 Sending to LLM for extraction...")
                extraction_result = extract_entities_from_text(
                    db, run, job, doc_text, pipeline_type, pipeline.config_json
                )
                
                # 5. Reconcile
                st.info("🔄 Reconciling against CRM...")
                reconcile_stats = reconcile_all(db, run, pipeline)
                
                # 6. Complete
                total_created = reconcile_stats.get("orgs_reconciled", 0) + reconcile_stats.get("people_reconciled", 0)
                run.run_status = "SUCCESS"
                run.completed_at = datetime.utcnow()
                run.records_processed = extraction_result["entities_extracted"]
                run.records_created = total_created
                
                pipeline.last_run_at = datetime.utcnow()
                pipeline.last_success_at = datetime.utcnow()
                
                db.commit()
                
                st.success(f"✅ Success! Processed {extraction_result['entities_extracted']} entities, created {total_created} CRM records.")
                
                # 7. Show summary
                entities = db.query(ExtractedEntity).filter_by(extraction_job_id=job.id).all()
                if entities:
                    st.subheader("Extracted Entities")
                    for e in entities:
                        payload = e.extracted_payload_json or {}
                        icon = "🏢" if e.entity_type == "ORGANIZATION" else "👤"
                        with st.expander(f"{icon} {e.raw_name} ({e.entity_type})"):
                            st.json(payload)
                
            except Exception as e:
                db.rollback()
                st.error(f"Pipeline failed: {e}")
                run_failed = db.query(PipelineRun).get(run.id)
                if run_failed:
                    run_failed.run_status = "FAILED"
                    run_failed.error_message = str(e)
                    run_failed.completed_at = datetime.utcnow()
                    db.commit()
