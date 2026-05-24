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
        st.session_state["start_processing"] = True
        st.session_state["upload_file_name"] = uploaded_file.name
        st.session_state["upload_file_bytes"] = uploaded_file.read()
        st.session_state["upload_pdf_method"] = pdf_method
        st.session_state["upload_custom_instructions"] = custom_instructions
        st.session_state["upload_linked_org_id"] = linked_org_id

if st.session_state.get("start_processing"):
    st.session_state["start_processing"] = False  # Clear to prevent infinite loops
    
    file_name = st.session_state["upload_file_name"]
    file_bytes = st.session_state["upload_file_bytes"]
    pdf_method = st.session_state["upload_pdf_method"]
    custom_instructions = st.session_state["upload_custom_instructions"]
    linked_org_id = st.session_state["upload_linked_org_id"]
    
    if True: # Keep indentation
        with st.spinner("Processing document..."):
            # 1. Read file content
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
                    
                    import gc
                    gc.collect()
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

            # 3. Create DB records and start background ingestion
            content_hash = hashlib.sha256(text_content.encode()).hexdigest()
            
            from market_comps.ingestion.async_processor import start_document_ingestion
            
            try:
                run_id = start_document_ingestion(
                    db=db,
                    text_content=text_content,
                    file_name=file_name,
                    storage_path=storage_path,
                    content_hash=content_hash,
                    final_instructions=final_instructions,
                    transcription_usage=transcription_usage
                )
                st.session_state.last_doc_upload_run_id = run_id
                st.success(f"🚀 Document processing started in the background (Run ID: {run_id}).")
                st.rerun()
            except Exception as e:
                db.rollback()
                st.error(f"Failed to start document ingestion: {e}")

# --- Display Results Outside of Submit Block ---
if "last_doc_upload_run_id" in st.session_state:
    last_run = db.query(IngestionRun).get(st.session_state.last_doc_upload_run_id)
    if last_run:
        st.divider()
        st.subheader(f"Latest Upload Results (Run ID: {last_run.id})")
        
        # Status Banner
        if last_run.run_status == "RUNNING":
            st.info("⏳ Processing in background... Please wait.")
            import time
            time.sleep(3)
            st.rerun()
        elif last_run.run_status == "SUCCESS":
            st.success("✅ Processing complete!")
        elif last_run.run_status == "FAILED":
            st.error(f"❌ Processing failed: {last_run.error_message}")
            
        # Refresh button if user wants to force refresh
        if st.button("Refresh Status"):
            st.rerun()
        
        # Look up document class
        source_doc = db.query(SourceDocument).filter_by(ingestion_run_id=last_run.id).first()
        if source_doc and source_doc.document_class:
            st.info(f"🏷️ Document classified as: **{source_doc.document_class}**")
            
        jobs = db.query(ExtractionJob).filter_by(ingestion_run_id=last_run.id).all()
        for job in jobs:
            entities = db.query(ExtractedEntity).filter_by(extraction_job_id=job.id).all()
            if entities:
                st.subheader(f"Extracted Entities ({job.schema_name})")
                for e in entities:
                    payload = e.extracted_payload_json or {}
                    icon = "🏢" if e.entity_type == "ORGANIZATION" else "👤"
                    with st.expander(f"{icon} {e.raw_name} ({e.entity_type})"):
                        st.json(payload)
