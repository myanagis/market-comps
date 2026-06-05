import streamlit as st
import logging
import io
import pandas as pd
from market_comps.config import settings, MODEL_OPTIONS, DEFAULT_LLM_MODEL, DEFAULT_MODELS, DEFAULT_SUMMARY_MODEL
from market_comps.ui import inject_global_style
from market_comps.market_intelligence_pipeline.flow_main import run_market_intelligence_pipeline

st.set_page_config(
    page_title="Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_global_style()

st.markdown("""
<style>
.section-header { color: #cbd5e1; font-size: 1.1rem; font-weight: 600; border-bottom: 1px solid #334155; padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0; }
.metric-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1rem 1.2rem; text-align: center; }
.metric-card .label { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem; }
.metric-card .value { color: #e2e8f0; font-size: 1.5rem; font-weight: 700; }
.metric-card .sub   { color: #64748b; font-size: 0.7rem; margin-top: 0.1rem; }
.usage-badge { background: #0f2744; border: 1px solid #1e4a7a; border-radius: 8px; padding: 0.6rem 0.8rem; font-size: 0.78rem; color: #93c5fd; margin-top: 0.8rem; }
.usage-badge b { color: #bfdbfe; }
.info-box { background: #1e293b; border-left: 4px solid #63b3ed; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;}
.lookback-badge { background: #334155; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.8rem; margin-right: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📊 <span class="accent">Market</span> <span style="color:#64748b">Intelligence</span></h1>
    <p>Comprehensive market research including M&A, Fundraising, IPOs, and Public Comps powered by intelligent pipelines and AI agents.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <span class="lookback-badge">🤝 <b>M&A:</b> Last 36 months</span>
    <span class="lookback-badge">💸 <b>Fundraising:</b> Last 24 months</span>
    <span class="lookback-badge">🚀 <b>IPOs:</b> Last 5 years</span>
    <span class="lookback-badge">📈 <b>Public Comps:</b> Current</span>
</div>
""", unsafe_allow_html=True)

if "mi_result" not in st.session_state:
    st.session_state["mi_result"] = None
if "mi_logs" not in st.session_state:
    st.session_state["mi_logs"] = ""

col_q, col_d = st.columns([1, 2])
with col_q:
    query = st.text_input(
        "Target Market / Company",
        placeholder="e.g. 'Cloud Security' or 'Stripe'",
    )
with col_d:
    description = st.text_input(
        "Description (Optional)",
        placeholder="Brief description to guide the LLM search",
    )

def format_model(m: str) -> str:
    in_price, out_price = settings.get_model_pricing(m)
    return f"{m} (${in_price:.2f} / ${out_price:.2f})"

with st.expander("⚙️ Advanced Pipeline Options", expanded=False):
    discovery_models = st.multiselect(
        "Discovery Models (Run in parallel)",
        options=MODEL_OPTIONS,
        default=DEFAULT_MODELS[:3],
        format_func=format_model,
        max_selections=3
    )
    col_p, col_v = st.columns(2)
    processing_model = col_p.selectbox("Processing Model (Deduplication)", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_SUMMARY_MODEL))
    verification_model = col_v.selectbox("Verification Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_LLM_MODEL))

run_clicked = st.button("🔍 Run Market Intelligence Pipeline", type="primary", disabled=not query.strip(), use_container_width=True)

if run_clicked and query.strip():
    status = st.status("🔄 Running Market Intelligence pipeline... (This involves multiple LLMs and takes up to 60 seconds)")
    
    def on_progress(msg: str):
        status.update(label=msg)

    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    
    loggers_to_capture = ["prefect", "prefect.TaskRun", "prefect.FlowRun", "market_comps"]
    for lname in loggers_to_capture:
        l = logging.getLogger(lname)
        l.setLevel(logging.INFO)
        l.addHandler(ch)

    try:
        result = run_market_intelligence_pipeline(
            query=query.strip(),
            description=description.strip(),
            discovery_models=discovery_models,
            processing_model=processing_model,
            verification_model=verification_model,
            progress_callback=on_progress
        )
        st.session_state["mi_result"] = result
        status.update(label="✅ Pipeline Completed!", state="complete")
    except Exception as e:
        status.update(label=f"❌ Pipeline error: {e}", state="error")
        st.error(f"Pipeline error: {e}")
    finally:
        for lname in loggers_to_capture:
            logging.getLogger(lname).removeHandler(ch)
        st.session_state["mi_logs"] = log_capture_string.getvalue()

result = st.session_state.get("mi_result")
if result is not None:
    for err in result.errors:
        st.warning(f"⚠️ {err}")
        
    u = result.usage
    st.markdown(f'<div class="usage-badge"><b>Pipeline Execution Metrics</b> &nbsp;&bull;&nbsp; <b>Calls:</b> {u.call_count} &nbsp;&bull;&nbsp; <b>Tokens:</b> {u.total_tokens:,} &nbsp;&bull;&nbsp; <b>Est. Cost:</b> ${u.estimated_cost_usd:.5f}</div>', unsafe_allow_html=True)

    data = result.data
    st.markdown(f"### Industry Classification: {data.get('industry_classification', 'N/A')}")
    
    tab_ma, tab_fund, tab_ipo, tab_comp, tab_logs = st.tabs(["🤝 M&A Events", "💸 Fundraising", "🚀 IPOs", "📈 Public Comps", "📋 Pipeline Logs"])
    
    with tab_ma:
        st.markdown('<div class="section-header">Recent M&A (36 months)</div>', unsafe_allow_html=True)
        ma_events = data.get("ma_events", [])
        if ma_events:
            st.dataframe(pd.DataFrame(ma_events), use_container_width=True, hide_index=True)
        else:
            st.info("No M&A events found.")
            
    with tab_fund:
        st.markdown('<div class="section-header">Recent Fundraising (24 months)</div>', unsafe_allow_html=True)
        fund_events = data.get("fundraising_events", [])
        if fund_events:
            st.dataframe(pd.DataFrame(fund_events), use_container_width=True, hide_index=True)
        else:
            st.info("No fundraising events found.")
            
    with tab_ipo:
        st.markdown('<div class="section-header">Recent IPOs (5 years)</div>', unsafe_allow_html=True)
        ipo_events = data.get("ipo_events", [])
        if ipo_events:
            st.dataframe(pd.DataFrame(ipo_events), use_container_width=True, hide_index=True)
        else:
            st.info("No IPO events found.")
            
    with tab_comp:
        st.markdown('<div class="section-header">Public Comparables</div>', unsafe_allow_html=True)
        comps = data.get("public_comps", [])
        if comps:
            # Flatten the live_metrics
            flat_comps = []
            for c in comps:
                row = c.copy()
                if "live_metrics" in row:
                    lm = row.pop("live_metrics")
                    row.update(lm)
                flat_comps.append(row)
            st.dataframe(pd.DataFrame(flat_comps), use_container_width=True, hide_index=True)
        else:
            st.info("No public comps found.")
            
    with tab_logs:
        st.markdown('<div class="section-header">Pipeline Execution Logs</div>', unsafe_allow_html=True)
        if st.session_state["mi_logs"]:
            st.code(st.session_state["mi_logs"], language="text")
        else:
            st.info("No logs available.")
