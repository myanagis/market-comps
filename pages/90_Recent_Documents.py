import streamlit as st
from sqlalchemy.orm import joinedload
from market_comps.db.session import get_db
from market_comps.db.models import SourceDocument, PipelineRun, ExtractionJob, ExtractedEntity, EntityMatch, Organization, FundProfile, Person
from market_comps.utils import format_est_datetime
from market_comps.ui import apply_theme

st.set_page_config(layout="wide", page_title="Recent Documents", page_icon="📄")
apply_theme()

st.title("📄 Recent Documents (Admin)")
st.write("View the 100 most recently ingested source documents and trace the exact database records they generated.")

db = next(get_db())

# Fetch last 100 docs
docs = db.query(SourceDocument).order_by(SourceDocument.created_at.desc()).limit(100).all()

if not docs:
    st.info("No documents found in the database.")
else:
    for doc in docs:
        tz_time = format_est_datetime(doc.created_at)
        doc_name = "SEC Form D" if doc.document_type == "SEC_XML" else doc.document_type
        
        # Determine URL
        from market_comps.config import get_supabase_url
        signed_url = get_supabase_url(doc.file_path) if doc.file_path else ""
        if doc.document_type == "SEC_XML":
            url_display = f"[{doc_name}]({doc.source_url})"
        elif signed_url:
            url_display = f"{doc.source_url} [(View)]({signed_url})"
        else:
            url_display = f"[{doc.source_url}]({doc.source_url})" if str(doc.source_url).startswith("http") else doc.source_url
            
        with st.expander(f"**{doc_name}** | {doc.source_url.split('/')[-1]} | Processed: {tz_time}"):
            st.markdown(f"**Source URL:** {url_display}")
            st.write(f"**Document Type:** {doc.document_type}")
            st.write(f"**Processed Date:** {tz_time}")
            
            st.divider()
            st.markdown("#### Extraction Jobs & Entities Generated")
            
            if not doc.pipeline_run_id:
                st.warning("No Pipeline Run ID linked to this document.")
                continue
                
            jobs = db.query(ExtractionJob).filter_by(pipeline_run_id=doc.pipeline_run_id).all()
            if not jobs:
                st.info("No extraction jobs found for this document's pipeline run.")
                continue
                
            for job in jobs:
                st.markdown(f"**Schema:** `{job.schema_name}` (Status: {job.status})")
                
                entities = db.query(ExtractedEntity).filter_by(extraction_job_id=job.id).all()
                if not entities:
                    st.caption("No entities extracted in this job.")
                    continue
                    
                for ent in entities:
                    st.markdown(f"- Extracted `{ent.entity_type}`: **{ent.raw_name}**")
                    
                    matches = db.query(EntityMatch).filter_by(extracted_entity_id=ent.id).all()
                    if matches:
                        for m in matches:
                            canonical_type = m.canonical_entity_type
                            canonical_id = m.canonical_entity_id
                            
                            # Resolve actual name if possible
                            resolved_name = "Unknown"
                            if canonical_type == "Organization":
                                org = db.query(Organization).filter_by(id=int(canonical_id)).first()
                                resolved_name = org.name if org else "Deleted Organization"
                            elif canonical_type == "FundProfile":
                                fund = db.query(FundProfile).filter_by(id=int(canonical_id)).first()
                                resolved_name = fund.fund_name if fund else "Deleted Fund"
                            elif canonical_type == "Person":
                                p = db.query(Person).filter_by(id=canonical_id).first()
                                resolved_name = p.full_name if p else "Deleted Person"
                                
                            st.markdown(f"  - 🔗 Linked to CRM Record: **{canonical_type}** (`{resolved_name}`)")
                    else:
                        st.caption("  - ⚠️ Not linked to any CRM records.")
                        
                with st.popover("View Raw Extracted JSON"):
                    for ent in entities:
                        st.write(f"Entity: {ent.raw_name}")
                        st.json(ent.extracted_payload_json)
