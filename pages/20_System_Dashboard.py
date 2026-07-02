import streamlit as st
import pandas as pd
from market_comps.db.session import SessionLocal
from market_comps.db.models import PipelineRun, CompanyAugmentationReport, Organization

st.set_page_config(page_title="System Dashboard", page_icon="⚙️", layout="wide")

st.title("⚙️ System Dashboard")
st.markdown("Monitor API usage, token consumption, and pipeline costs across the system.")

db = SessionLocal()
try:
    # Query all pipeline runs that have associated augmentation reports
    # A simpler query in python is done below, we can remove the complex sqlalchemy query that was unused.
    
    runs = db.query(PipelineRun).order_by(PipelineRun.id.desc()).all()
    
    data = []
    total_llm_tokens = 0
    total_llm_cost = 0.0
    total_exa_calls = 0
    total_exa_cost = 0.0
    
    for run in runs:
        # Get associated company name if this was an augmentation run
        report = db.query(CompanyAugmentationReport).filter_by(pipeline_run_id=run.id).first()
        company_name = "System Pipeline"
        if report and report.organization_id:
            org = db.query(Organization).filter_by(id=report.organization_id).first()
            if org:
                company_name = org.name
                
        t = run.llm_total_tokens or 0
        lc = run.llm_estimated_cost_usd or 0.0
        ecalls = run.exa_calls or 0
        ecost = run.exa_estimated_cost_usd or 0.0
        
        total_llm_tokens += t
        total_llm_cost += lc
        total_exa_calls += ecalls
        total_exa_cost += ecost
        
        # Format date safely
        date_str = ""
        # if started_at is missing, use id or created_at if we have timestamp mixin (Wait, pipeline runs don't have created_at natively unless mixed in).
        # We can just show ID for sorting.
        
        data.append({
            "Run ID": run.id,
            "Target Company": company_name,
            "Status": run.run_status,
            "LLM Tokens": f"{t:,}",
            "LLM Cost": f"${lc:.4f}",
            "Exa Calls": ecalls,
            "Exa Cost": f"${ecost:.4f}",
            "Total Cost": f"${(lc + ecost):.4f}"
        })
        
    # Top level metrics
    st.subheader("Global System Usage (All Time)")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total LLM Tokens", f"{total_llm_tokens:,}")
    col2.metric("Total LLM Cost", f"${total_llm_cost:.2f}")
    col3.metric("Total Exa Calls", f"{total_exa_calls:,}")
    col4.metric("Total Exa Cost", f"${total_exa_cost:.2f}")
    col5.metric("Total System Cost", f"${(total_llm_cost + total_exa_cost):.2f}")
    
    st.divider()
    
    st.subheader("Pipeline Run History")
    st.markdown("Chronological list of pipeline executions and their individual costs.")
    
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No pipeline runs recorded yet.")
        
finally:
    db.close()
