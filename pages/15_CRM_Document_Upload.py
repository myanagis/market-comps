import streamlit as st
import hashlib
from datetime import datetime
import pdfplumber
import io

from market_comps.db.session import get_db
from market_comps.db.models import (
    Pipeline, IngestionRun, SourceDocument, DocumentText, ExtractionJob, ExtractedEntity, ExtractedRelationship, Organization
)
from market_comps.ingestion.extractor import extract_entities_from_text
from market_comps.ingestion.reconciler import reconcile_all
from market_comps.config import supabase_client

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
    st.info("💡 The system will automatically classify your document and run the recommended extraction schemas.")
    
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
    
    col3, col4 = st.columns(2)
    pdf_method = col3.selectbox("PDF Parsing Method", ["vlm_plus_text", "text", "vlm", "ocr", "paddle_ocr"], index=0, format_func=lambda x: {
        "vlm_plus_text": "Hybrid (Native Text + VLM)",
        "text": "Native Text (Fast, Free)",
        "vlm": "VLM Vision-based",
        "ocr": "Mistral OCR ($2 / 1k pages)",
        "paddle_ocr": "Paddle OCR (Local, Free)"
    }.get(x, x))
    
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
            transcription_usage = None
            
            if file_name.lower().endswith(".pdf"):
                from market_comps.document_pipeline.tasks_transcription import transcribe_document
                from market_comps.config import DEFAULT_LLM_MODEL
                try:
                    st.info(f"📄 Extracting text using '{pdf_method}' method...")
                    text_content, transcription_usage, raw_texts = transcribe_document(
                        pdf_bytes=file_bytes, 
                        filename=file_name, 
                        method=pdf_method, 
                        model=DEFAULT_LLM_MODEL
                    )
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

            # 1.5 Upload to Supabase Storage
            storage_path = None
            if supabase_client:
                st.info("☁️ Uploading document to Supabase Storage...")
                try:
                    from market_comps.config import settings
                    # Generate a unique path: timestamp + file_name
                    path = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file_name}"
                    supabase_client.storage.from_(settings.supabase_storage_bucket).upload(
                        file=file_bytes,
                        path=path,
                        file_options={"content-type": "application/pdf" if file_name.lower().endswith(".pdf") else "text/plain"}
                    )
                    storage_path = path
                except Exception as e:
                    from market_comps.config import settings
                    st.warning(f"Failed to upload to Supabase Storage: {e}. Ensure the '{settings.supabase_storage_bucket}' bucket exists and credentials are correct.")

            # 2. Build instructions context
            final_instructions = custom_instructions
            if linked_org_id != 0:
                linked_org = db.query(Organization).get(linked_org_id)
                if linked_org:
                    ctx_note = f"NOTE: This document is related to the organization '{linked_org.name}'. Pay special attention to their properties and relationships."
                    final_instructions = f"{ctx_note}\n{final_instructions}".strip()

            # 3. Create DB records
            content_hash = hashlib.sha256(text_content.encode()).hexdigest()
            
            run = IngestionRun(
                pipeline_id=None,
                run_status="RUNNING",
                started_at=datetime.utcnow()
            )
            db.add(run)
            db.flush()
            
            source_doc = SourceDocument(
                ingestion_run_id=run.id,
                document_type="PDF" if file_name.lower().endswith(".pdf") else "TEXT",
                source_url=file_name,
                file_path=storage_path,
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
            
            try:
                # 3.5 Classify Document
                st.info("🔍 Classifying document...")
                from market_comps.ingestion.classifier import classify_document, get_recommended_schemas
                from market_comps.llm_client import LLMClient
                
                try:
                    class_result, class_usage = classify_document(text_content, LLMClient())
                    doc_class = class_result.get("document_type", "unknown")
                    source_doc.document_class = doc_class
                    source_doc.classification_result_json = class_result
                    db.flush()
                    st.success(f"🏷️ Classified as: **{doc_class}** (Confidence: {class_result.get('confidence', 0):.2f})")
                    
                    schemas_to_run = get_recommended_schemas(doc_class)
                    if not schemas_to_run:
                        st.warning(f"No schemas recommended for class '{doc_class}'. Running default extraction.")
                        schemas_to_run = ["DOCUMENT_ENTITIES"]
                except Exception as e:
                    st.error(f"Classification failed: {e}. Falling back to default schema.")
                    source_doc.document_class = "unknown"
                    source_doc.classification_result_json = {"error": str(e)}
                    db.flush()
                    schemas_to_run = ["DOCUMENT_ENTITIES"]
                    class_usage = None
                
                # 4. Run LLM Extraction for each schema
                total_entities_extracted = 0
                all_jobs = []
                total_usage_dict = {
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "call_count": 0,
                    "traces": []
                }
                
                for schema_idx, schema_name in enumerate(schemas_to_run):
                    st.info(f"🤖 [{schema_idx+1}/{len(schemas_to_run)}] Running schema: **{schema_name}**...")
                    
                    job = ExtractionJob(
                        ingestion_run_id=run.id,
                        document_text_id=doc_text.id,
                        schema_name=schema_name,
                        status="IN_PROGRESS",
                        started_at=datetime.utcnow()
                    )
                    db.add(job)
                    db.flush()
                    all_jobs.append(job)
                    
                    try:
                        extraction_result = extract_entities_from_text(
                            db, run, job, doc_text, schema_name, {"llm_instruction": final_instructions}
                        )
                        
                        total_entities_extracted += extraction_result["entities_extracted"]
                        
                        job_usage = extraction_result["llm_usage"]
                        
                        if schema_idx == 0:
                            from market_comps.models import LLMUsage
                            first_usage = LLMUsage(**job_usage)
                            if transcription_usage:
                                first_usage.total_prompt_tokens += transcription_usage.total_prompt_tokens
                                first_usage.total_completion_tokens += transcription_usage.total_completion_tokens
                                first_usage.total_tokens += transcription_usage.total_tokens
                                first_usage.estimated_cost_usd += transcription_usage.estimated_cost_usd
                                first_usage.call_count += transcription_usage.call_count
                                first_usage.traces = transcription_usage.traces + first_usage.traces
                            if class_usage:
                                first_usage.total_prompt_tokens += class_usage.total_prompt_tokens
                                first_usage.total_completion_tokens += class_usage.total_completion_tokens
                                first_usage.total_tokens += class_usage.total_tokens
                                first_usage.estimated_cost_usd += class_usage.estimated_cost_usd
                                first_usage.call_count += class_usage.call_count
                                first_usage.traces = class_usage.traces + first_usage.traces
                            job_usage = first_usage.model_dump()
                            
                        total_usage_dict["total_prompt_tokens"] += job_usage.get("total_prompt_tokens", 0)
                        total_usage_dict["total_completion_tokens"] += job_usage.get("total_completion_tokens", 0)
                        total_usage_dict["total_tokens"] += job_usage.get("total_tokens", 0)
                        total_usage_dict["estimated_cost_usd"] += job_usage.get("estimated_cost_usd", 0.0)
                        total_usage_dict["call_count"] += job_usage.get("call_count", 0)
                        total_usage_dict["traces"] += job_usage.get("traces", [])
                        
                        job.llm_usage_json = job_usage
                        job.status = "SUCCESS"
                        job.completed_at = datetime.utcnow()
                        db.flush()
                    except Exception as e:
                        job.status = "FAILED"
                        job.error_message = str(e)
                        job.completed_at = datetime.utcnow()
                        db.flush()
                        st.error(f"Schema {schema_name} failed: {e}")
                
                # 5. Reconcile
                st.info("🔄 Reconciling combined results against CRM...")
                reconcile_stats = reconcile_all(db, run, None)
                
                # 6. Complete
                total_created = reconcile_stats.get("orgs_reconciled", 0) + reconcile_stats.get("people_reconciled", 0)
                run.run_status = "SUCCESS"
                run.completed_at = datetime.utcnow()
                run.records_processed = total_entities_extracted
                run.records_created = total_created
                
                db.commit()
                
                st.success(f"✅ Success! Processed {total_entities_extracted} entities across {len(schemas_to_run)} schemas, created {total_created} CRM records.")
                
                # 7. Show summary
                for job in all_jobs:
                    entities = db.query(ExtractedEntity).filter_by(extraction_job_id=job.id).all()
                    if entities:
                        st.subheader(f"Extracted Entities ({job.schema_name})")
                        for e in entities:
                            payload = e.extracted_payload_json or {}
                            icon = "🏢" if e.entity_type == "ORGANIZATION" else "👤"
                            with st.expander(f"{icon} {e.raw_name} ({e.entity_type})"):
                                st.json(payload)
                
            except Exception as e:
                db.rollback()
                st.error(f"Pipeline failed: {e}")
                run_failed = db.query(IngestionRun).get(run.id)
                if run_failed:
                    run_failed.run_status = "FAILED"
                    run_failed.error_message = str(e)
                    run_failed.completed_at = datetime.utcnow()
                    db.commit()
