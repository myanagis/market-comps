import streamlit as st
import pandas as pd
from market_comps.db.session import get_db
from market_comps.db.models import (
    SourceDocument, PipelineRun, Pipeline, DocumentText, 
    ExtractionJob, ExtractedEntity, ExtractedRelationship,
    CanonicalMutation, EntityMatch
)
from market_comps.config import supabase_client

st.set_page_config(page_title="Data Provenance Inspector", page_icon="🔬", layout="wide")
st.title("🔬 Data Provenance Inspector")
st.markdown("Deep dive into the backend processing and provenance of imported documents.")

try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

docs = db.query(SourceDocument).join(PipelineRun).join(Pipeline).order_by(SourceDocument.created_at.desc()).all()
if not docs:
    st.warning("No imported documents found.")
    st.stop()

doc_opts = {d.id: f"{d.source_url} (Pipeline Run #{d.pipeline_run_id}) [Created: {d.created_at.strftime('%Y-%m-%d %H:%M')}]" for d in docs}
selected_doc_id = st.selectbox("Select Source Document", options=list(doc_opts.keys()), format_func=lambda x: doc_opts[x])

if selected_doc_id:
    doc = db.query(SourceDocument).get(selected_doc_id)
    run = db.query(PipelineRun).get(doc.pipeline_run_id)
    
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["Raw Document", "Extracted Payload", "Entity Matches", "CRM Mutations"])
    
    with tab1:
        st.subheader("Raw File Viewer")
        if doc.file_path and supabase_client:
            try:
                # Generate signed URL valid for 60 seconds
                res = supabase_client.storage.from_("documents").create_signed_url(doc.file_path, 60)
                signed_url = res.get("signedURL") if isinstance(res, dict) else res
                if signed_url:
                    st.markdown(f"**Storage Path:** `{doc.file_path}`")
                    
                    if doc.file_path.lower().endswith(".pdf"):
                        st.components.v1.iframe(signed_url, height=800, scrolling=True)
                    else:
                        st.markdown(f"[Download / View Raw File]({signed_url})")
            except Exception as e:
                st.error(f"Could not load file from Supabase Storage: {e}")
        else:
            st.info("No raw file was uploaded to Supabase Storage for this document.")
            
        st.subheader("Extracted Text Content")
        dt = db.query(DocumentText).filter_by(source_document_id=doc.id).first()
        if dt:
            with st.expander("View Text Content Sent to LLM", expanded=not bool(doc.file_path)):
                st.text(dt.raw_content)
                
    with tab2:
        st.subheader("LLM Extracted Payloads")
        jobs = db.query(ExtractionJob).filter_by(pipeline_run_id=run.id).all()
        for job in jobs:
            st.markdown(f"**Job ID:** `{job.id}` | **Schema:** `{job.schema_name}` | **Status:** `{job.status}`")
            entities = db.query(ExtractedEntity).filter_by(extraction_job_id=job.id).all()
            if entities:
                st.write("**Extracted Entities:**")
                for e in entities:
                    with st.expander(f"{e.entity_type}: {e.raw_name}"):
                        st.json(e.extracted_payload_json)
            
            rels = db.query(ExtractedRelationship).filter_by(extraction_job_id=job.id).all()
            if rels:
                st.write("**Extracted Relationships:**")
                for r in rels:
                    st.json(r.relationship_payload_json)

    with tab3:
        st.subheader("Entity Matches")
        st.markdown("Shows how raw extracted names were mapped to canonical CRM IDs.")
        matches = db.query(EntityMatch).join(ExtractedEntity).join(ExtractionJob).filter(ExtractionJob.pipeline_run_id == run.id).all()
        if matches:
            for m in matches:
                st.write(f"- Extracted Entity **{m.extracted_entity.raw_name}** ({m.extracted_entity.entity_type}) matched to CRM **{m.canonical_entity_type}** `#{m.canonical_entity_id}` via `{m.match_method}` (Confidence: {m.match_confidence})")
        else:
            st.write("No matches found.")

    with tab4:
        st.subheader("CRM Canonical Mutations")
        st.markdown("Shows exactly which fields were created or updated in the CRM by this document.")
        muts = db.query(CanonicalMutation).join(ExtractionJob).filter(ExtractionJob.pipeline_run_id == run.id).all()
        if muts:
            df = pd.DataFrame([{
                "Entity Type": m.canonical_entity_type,
                "Entity ID": m.canonical_entity_id,
                "Mutation Type": m.mutation_type,
                "Field": m.field_name,
                "Old Value": m.old_value,
                "New Value": m.new_value,
            } for m in muts])
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No CRM mutations logged.")
