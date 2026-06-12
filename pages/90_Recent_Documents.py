import streamlit as st
from sqlalchemy.orm import joinedload
from market_comps.db.session import get_db
from market_comps.db.models import SourceDocument, PipelineRun, ExtractionJob, ExtractedEntity, EntityMatch, Organization, FundProfile, Person
from market_comps.utils import format_est_datetime
from market_comps.ui import inject_global_style

st.set_page_config(layout="wide", page_title="Recent Documents", page_icon="📄")
inject_global_style()

st.title("📄 Recent Documents (Admin)")
st.write("View the 100 most recently ingested source documents and trace the exact database records they generated.")

db = next(get_db())

# Fetch last 100 docs
docs = db.query(SourceDocument).order_by(SourceDocument.created_at.desc()).limit(100).all()

if not docs:
    st.info("No documents found in the database.")
else:
    import pandas as pd
    table_data = []
    
    for doc in docs:
        tz_time = format_est_datetime(doc.created_at)
        doc_name = "SEC Form D" if doc.document_type == "SEC_XML" else doc.document_type
        title = doc.title or "Unknown Title"
        
        extracted_summary = []
        linked_summary = []
        
        if doc.pipeline_run_id:
            jobs = db.query(ExtractionJob).filter_by(pipeline_run_id=doc.pipeline_run_id).all()
            for job in jobs:
                entities = db.query(ExtractedEntity).filter_by(extraction_job_id=job.id).all()
                for ent in entities:
                    extracted_summary.append(f"{ent.entity_type} ({ent.raw_name})")
                    
                    matches = db.query(EntityMatch).filter_by(extracted_entity_id=ent.id).all()
                    for m in matches:
                        canonical_type = m.canonical_entity_type
                        canonical_id = m.canonical_entity_id
                        resolved_name = "Unknown"
                        if canonical_type == "Organization":
                            org = db.query(Organization).filter_by(id=int(canonical_id)).first()
                            resolved_name = org.name if org else "Deleted"
                        elif canonical_type == "FundProfile":
                            fund = db.query(FundProfile).filter_by(id=int(canonical_id)).first()
                            resolved_name = fund.fund_name if fund else "Deleted"
                        elif canonical_type == "Person":
                            p = db.query(Person).filter_by(id=canonical_id).first()
                            resolved_name = p.full_name if p else "Deleted"
                        linked_summary.append(f"{canonical_type}: {resolved_name}")
                        
        table_data.append({
            "Title": title,
            "Type": doc_name,
            "Source URL": doc.source_url,
            "Processed Date": tz_time,
            "Extracted Entities": " | ".join(extracted_summary) if extracted_summary else "None",
            "Linked CRM Records": " | ".join(linked_summary) if linked_summary else "None"
        })
        
    df = pd.DataFrame(table_data)
    st.dataframe(
        df,
        column_config={
            "Source URL": st.column_config.LinkColumn("Source URL"),
        },
        use_container_width=True,
        hide_index=True
    )
