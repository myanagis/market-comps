# pages/1_Public_Comps.py
"""
Public Comps — Find publicly traded comparables using the Chorus of LLMs.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Public Comps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from market_comps.chorus_comps_engine import ChorusCompsEngine
from market_comps.config import settings, MODEL_OPTIONS
from market_comps.models import CompsResult, CompanyMetrics, ScanFilters

from market_comps.ui import inject_global_style, create_chorus_progress_status

inject_global_style()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1rem 1.2rem; text-align: center; }
.metric-card .label { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem; }
.metric-card .value { color: #e2e8f0; font-size: 1.5rem; font-weight: 700; }
.metric-card .sub   { color: #64748b; font-size: 0.7rem; margin-top: 0.1rem; }

.usage-badge { background: #0f2744; border: 1px solid #1e4a7a; border-radius: 8px; padding: 0.6rem 0.8rem; font-size: 0.78rem; color: #93c5fd; margin-top: 0.5rem; }
.usage-badge b { color: #bfdbfe; }

.section-header { color: #cbd5e1; font-size: 1.1rem; font-weight: 600; border-bottom: 1px solid #334155; padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0; }
.desc-text { color: #94a3b8; font-style: italic; font-size: 0.9rem; line-height: 1.6; }
.info-box { background: #1e293b; border-left: 4px solid #63b3ed; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; color: #94a3b8; font-size: 0.9rem; }

.how-it-works { color: #94a3b8; font-size: 0.88rem; line-height: 1.7; }
.how-it-works h4 { color: #cbd5e1; margin: 0.8rem 0 0.3rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_currency(val: Optional[float], decimals: int = 1) -> str:
    if val is None: return "—"
    if val >= 1e12: return f"${val/1e12:.{decimals}f}T"
    if val >= 1e9:  return f"${val/1e9:.{decimals}f}B"
    if val >= 1e6:  return f"${val/1e6:.{decimals}f}M"
    return f"${val:,.0f}"

def fmt_multiple(val: Optional[float], suffix: str = "x") -> str:
    return "—" if val is None else f"{val:.1f}{suffix}"

def fmt_pct(val: Optional[float]) -> str:
    return "—" if val is None else f"{val:.1f}%"

def build_dataframe(comps: list[CompanyMetrics]) -> pd.DataFrame:
    rows = []
    for m in comps:
        rows.append({
            "Company": m.name, "Ticker": m.ticker, "Exchange": m.exchange or "—",
            "Country": m.country or "—", "Sector": m.sector or "—",
            "Industry": m.industry or "—",
            "Market Cap": fmt_currency(m.market_cap_usd),
            "EV": fmt_currency(m.ev_usd),
            "Rev (TTM)": fmt_currency(m.revenue_ttm_usd),
            "Rev (NTM)": fmt_currency(m.revenue_ntm_usd),
            "EV/Rev (TTM)": fmt_multiple(m.ev_to_revenue_ttm),
            "EV/Rev (NTM)": fmt_multiple(m.ev_to_revenue_ntm),
            "Gross Margin": fmt_pct(m.gross_margin_pct),
            "EBITDA Margin": fmt_pct(m.ebitda_margin_pct),
            "Rev Growth YoY": fmt_pct(m.revenue_growth_yoy_pct),
            "Data": "✅" if m.data_available else "⚠️",
        })
    return pd.DataFrame(rows)

def build_export_dataframe(comps: list[CompanyMetrics]) -> pd.DataFrame:
    rows = []
    for m in comps:
        rows.append({
            "Company": m.name, "Ticker": m.ticker, "Exchange": m.exchange,
            "Country": m.country, "Sector": m.sector, "Industry": m.industry,
            "Description": m.description,
            "Market_Cap_USD": m.market_cap_usd, "EV_USD": m.ev_usd,
            "Revenue_TTM_USD": m.revenue_ttm_usd, "Revenue_NTM_USD": m.revenue_ntm_usd,
            "EV_Revenue_TTM": m.ev_to_revenue_ttm, "EV_Revenue_NTM": m.ev_to_revenue_ntm,
            "Gross_Margin_Pct": m.gross_margin_pct, "EBITDA_Margin_Pct": m.ebitda_margin_pct,
            "Revenue_Growth_YoY_Pct": m.revenue_growth_yoy_pct,
            "Data_Available": m.data_available, "Data_Notes": m.data_notes,
        })
    return pd.DataFrame(rows)

def generate_html_report(query: str, comps_result: CompsResult) -> str:
    from datetime import datetime
    now = datetime.now().strftime("%B %d, %Y %H:%M")

    def tbl(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "<p><em>No data.</em></p>"
        header = "".join(f"<th>{c}</th>" for c in df.columns)
        rows = "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
            for row in df.itertuples(index=False)
        )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"

    comps_html = ""
    if comps_result and comps_result.comps:
        comps_df = build_dataframe(comps_result.comps)
        comps_html = f"<h2>Public Market Comparables</h2>{tbl(comps_df)}"

    css = """
    body{font-family:'Segoe UI',Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#1e293b;background:#fff}
    h1{color:#1e40af;border-bottom:2px solid #3b82f6;padding-bottom:8px}
    h2{color:#1e293b;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:2rem}
    table{border-collapse:collapse;width:100%;margin:12px 0;font-size:0.88rem}
    th{background:#1e40af;color:#fff;padding:8px 10px;text-align:left}
    td{padding:7px 10px;border-bottom:1px solid #e2e8f0}
    tr:nth-child(even) td{background:#f8fafc}
    .meta{color:#94a3b8;font-size:0.82rem;margin-bottom:2rem}
    """
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
    <title>Public Comps: {query}</title>
    <style>{css}</style></head><body>
    <h1>Public Comps Report</h1>
    <p class="meta">Query: <strong>{query}</strong> &nbsp;&mdash;&nbsp; Generated: {now}</p>
    {comps_html}
    </body></html>"""

def _sortable(val: str) -> float:
    if val in ("—", ""): return float("-inf")
    s = val.replace("$","").replace(",","").replace("x","").replace("%","").strip()
    try:
        mult = 1
        if s.endswith("T"): mult, s = 1e12, s[:-1]
        elif s.endswith("B"): mult, s = 1e9, s[:-1]
        elif s.endswith("M"): mult, s = 1e6, s[:-1]
        return float(s) * mult
    except ValueError:
        return float("-inf")


# ── Session state ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state["result"] = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 <span class="accent">Public</span> <span style="color:#64748b">Comps</span></h1>
    <p>Discover publicly traded comparables with live market data for any company.</p>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
col_q, col_d, col_c = st.columns([1, 1, 1])
with col_q:
    query = st.text_input(
        "Company / industry",
        label_visibility="collapsed",
        placeholder="Company name or industry (e.g. 'Stripe', 'cloud ERP')",
        key="query_input",
    )
with col_d:
    description = st.text_input(
        "Company description (optional)",
        label_visibility="collapsed",
        placeholder="Brief description (e.g. 'B2B fintech payments API')",
        key="description_input",
    )
with col_c:
    competitors_to_include = st.text_input(
        "Competitors to include (optional)",
        label_visibility="collapsed",
        placeholder="Specific competitors to explicitly include",
        key="competitors_to_include",
    )

# ── Config ────────────────────────────────────────────────────────────────────
from market_comps.cross_checker.cross_checker import DEFAULT_MODELS as _CHORUS_DEFAULTS

def format_model(m: str) -> str:
    in_price, out_price = settings.get_model_pricing(m)
    return f"{m} (${in_price:.2f} / ${out_price:.2f})"

with st.expander("⚙️ Advanced Options", expanded=False):
    _ao1, _ao2 = st.columns([3, 1])
    with _ao1:
        chorus_models = st.multiselect(
            "Chorus models (queried in parallel for public comps)",
            options=MODEL_OPTIONS,
            default=_CHORUS_DEFAULTS,
            format_func=format_model,
        )
    with _ao2:
        n_comps = st.slider("Number of Comps", min_value=5, max_value=30, value=10, step=1)
    st.markdown("**Filters (Yahoo Finance execution)**")
    _fc1, _fc2 = st.columns(2)
    with _fc1:
        filter_countries = st.text_input("Countries (comma-separated)", "", placeholder="e.g. United States, Canada")
        filter_exchanges = st.text_input("Exchanges (comma-separated)", "", placeholder="e.g. NYSE, NASDAQ")
    with _fc2:
        filter_sectors = st.text_input("Sectors (comma-separated)", "", placeholder="e.g. Technology, Healthcare")
        filter_industries = st.text_input("Industries (comma-separated)", "", placeholder="e.g. Cloud Software")
    _mc1, _mc2 = st.columns(2)
    with _mc1:
        min_mc_b = st.number_input("Min Mkt Cap ($B)", min_value=0.0, value=0.0, step=0.5)
    with _mc2:
        max_mc_b = st.number_input("Max Mkt Cap ($B)", min_value=0.0, value=0.0, step=10.0)

# ── Single action button ──────────────────────────────────────────────────────
_b1, _bpad = st.columns([2, 6])
with _b1:
    run_clicked = st.button(
        "🔍 Find Public Comps",
        type="primary",
        disabled=not query.strip() or not chorus_models,
        use_container_width=True,
    )

if st.session_state.get("result") is None and not run_clicked:
    st.markdown("""
    <div class="info-box">
        👆 Enter a <b>company name</b> or <b>industry</b> and click <b>Find Public Comps</b>.<br><br>
        The Chorus of LLMs will discover
        <b>publicly traded comparables</b> matching your description and load live market metrics via Yahoo Finance.
    </div>
    """, unsafe_allow_html=True)

def _split(s): return [x.strip() for x in s.split(",") if x.strip()]

# ── Run on button click ────────────────────────────────────────
if run_clicked and query.strip() and chorus_models:
    st.session_state["result"] = None

    filters = ScanFilters(
        countries=_split(filter_countries), exchanges=_split(filter_exchanges),
        sectors=_split(filter_sectors), industries=_split(filter_industries),
        min_market_cap_usd=min_mc_b * 1e9 if min_mc_b > 0 else None,
        max_market_cap_usd=max_mc_b * 1e9 if max_mc_b > 0 else None,
    )

    status_comps, on_done_comps = create_chorus_progress_status(
        status_label=f"⏳ Querying {len(chorus_models)} models for public comparable tickers…",
        models=chorus_models,
    )
    with status_comps:
        try:
            engine = ChorusCompsEngine(models=chorus_models)
            result = engine.find_comps(
                company=query.strip(),
                description=description.strip(),
                competitors_to_include=competitors_to_include.strip(),
                filters=filters,
                limit=n_comps,
                on_model_complete=on_done_comps,
            )
            st.session_state["result"] = result
            status_comps.update(label=f"✅ Public Comps done", state="complete", expanded=False)
        except Exception as exc:
            status_comps.update(label=f"❌ Public Comps error: {exc}", state="error")
            st.error(f"❌ Public Comps error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Results: Public Comps
# ══════════════════════════════════════════════════════════════════════════════
result: Optional[CompsResult] = st.session_state.get("result")

if result is not None:
    if result.errors:
        for err in result.errors:
            st.warning(f"⚠️ {err}")

    comps = result.comps
    n_with_data = sum(1 for c in comps if c.data_available)

    st.markdown('<div class="section-header">📊 Public Comps</div>', unsafe_allow_html=True)
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        st.markdown(f'<div class="metric-card"><div class="label">Query</div><div class="value" style="font-size:1rem;">{result.query}</div></div>', unsafe_allow_html=True)
    with kc2:
        st.markdown(f'<div class="metric-card"><div class="label">Comps Found</div><div class="value">{len(comps)}</div><div class="sub">{result.candidates_found} candidates scanned</div></div>', unsafe_allow_html=True)
    with kc3:
        st.markdown(f'<div class="metric-card"><div class="label">Data Coverage</div><div class="value">{n_with_data}/{len(comps)}</div><div class="sub">tickers with live data</div></div>', unsafe_allow_html=True)
    with kc4:
        u = result.llm_usage
        st.markdown(f'<div class="metric-card"><div class="label">LLM Cost</div><div class="value">${u.estimated_cost_usd:.4f}</div><div class="sub">{u.total_tokens:,} tokens · {u.call_count} calls</div></div>', unsafe_allow_html=True)

    if result.llm_usage.call_count > 0:
        _u = result.llm_usage
        st.markdown(
            f'<div class="usage-badge"><b>Model:</b> {result.model_used}&nbsp;&nbsp;'
            f'<b>API Calls:</b> {_u.call_count}&nbsp;&nbsp;'
            f'<b>Prompt:</b> {_u.total_prompt_tokens:,}&nbsp;&nbsp;'
            f'<b>Completion:</b> {_u.total_completion_tokens:,}&nbsp;&nbsp;'
            f'<b>Total:</b> {_u.total_tokens:,}&nbsp;&nbsp;'
            f'<b>Est. cost:</b> ${_u.estimated_cost_usd:.5f}</div>',
            unsafe_allow_html=True,
        )

    if not comps:
        st.markdown('<div class="info-box">No comparable companies found. Try a different query or adjust filters.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header">📋 Comparable Companies</div>', unsafe_allow_html=True)
        sort_col = st.selectbox(
            "Sort by", ["EV/Rev (TTM)", "EV/Rev (NTM)", "Market Cap", "Gross Margin", "Rev Growth YoY", "Company"],
            index=0, key="sort_col", label_visibility="collapsed",
        )
        df = build_dataframe(comps)
        df_display = df.copy()
        if sort_col != "Company":
            sort_vals = df_display[sort_col].apply(_sortable)
            df_display = df_display.iloc[sort_vals.argsort()[::-1].values]
        st.dataframe(df_display, use_container_width=True, hide_index=True,
                     column_config={
                         "Data": st.column_config.TextColumn("Data", width=60),
                         "Ticker": st.column_config.TextColumn("Ticker", width=90),
                         "Exchange": st.column_config.TextColumn("Exchange", width=90),
                         "Country": st.column_config.TextColumn("Country", width=100),
                     })

        st.markdown('<div class="section-header">🏢 Company Details</div>', unsafe_allow_html=True)
        for m in comps:
            with st.expander(f"**{m.name}** ({m.ticker}) — {m.sector or 'N/A'}"):
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.markdown(f"**Exchange:** {m.exchange or '—'}")
                    st.markdown(f"**Country:** {m.country or '—'}")
                    st.markdown(f"**Sector:** {m.sector or '—'}")
                    st.markdown(f"**Industry:** {m.industry or '—'}")
                with dc2:
                    st.markdown(f"**Market Cap:** {fmt_currency(m.market_cap_usd)}")
                    st.markdown(f"**EV:** {fmt_currency(m.ev_usd)}")
                    st.markdown(f"**Rev (TTM):** {fmt_currency(m.revenue_ttm_usd)}")
                    st.markdown(f"**Rev (NTM):** {fmt_currency(m.revenue_ntm_usd)}")
                with dc3:
                    st.markdown(f"**EV/Rev (TTM):** {fmt_multiple(m.ev_to_revenue_ttm)}")
                    st.markdown(f"**EV/Rev (NTM):** {fmt_multiple(m.ev_to_revenue_ntm)}")
                    st.markdown(f"**Gross Margin:** {fmt_pct(m.gross_margin_pct)}")
                    st.markdown(f"**EBITDA Margin:** {fmt_pct(m.ebitda_margin_pct)}")
                if m.description:
                    st.markdown(f'<div class="desc-text">{m.description}</div>', unsafe_allow_html=True)
                if not m.data_available:
                    st.warning(f"⚠️ Data unavailable: {m.data_notes}")

        st.markdown('<div class="section-header">⬇️ Export</div>', unsafe_allow_html=True)
        export_df = build_export_dataframe(comps)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        safe_query = "".join(c if c.isalnum() else "_" for c in result.query)[:40]
        st.download_button("📥 Download CSV", data=csv_bytes,
                           file_name=f"public_comps_{safe_query}.csv", mime="text/csv", type="secondary")

# ══════════════════════════════════════════════════════════════════════════════
# Download Report
# ══════════════════════════════════════════════════════════════════════════════
_pr = st.session_state.get("result")
if _pr is not None:
    st.markdown('<div class="section-header">⬇️ Download Report</div>', unsafe_allow_html=True)
    _report_query = query.strip() or "report"
    html_bytes = generate_html_report(_report_query, _pr).encode("utf-8")
    safe_q = "".join(c if c.isalnum() else "_" for c in _report_query)[:40]
    col_dl1, col_dl2 = st.columns([2, 6])
    with col_dl1:
        st.download_button(
            "📄 Download HTML Report",
            data=html_bytes,
            file_name=f"public_comps_report_{safe_q}.html",
            mime="text/html",
            type="primary",
            use_container_width=True,
            help="Opens in any browser. Use File → Print → Save as PDF.",
        )

# ══════════════════════════════════════════════════════════════════════════════
# How It Works
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🤖 How this page works", expanded=False):
    st.markdown("""
<div class="how-it-works">

<h4>Find Public Comps</h4>
Five LLMs suggest public comparable tickers in parallel, a deduplication LLM reconciles them,
and then live financial data is fetched from <b>yfinance</b> (market cap, EV, revenue TTM/NTM, margins, growth).

<h4>Sources and Accuracy</h4>
Market metrics come directly from <b>Yahoo Finance</b> (real-time).
</div>
""", unsafe_allow_html=True)
    
    st.markdown("#### LLM Prompts Used")
    from market_comps.chorus_comps_engine import _COMPS_QUESTION_TEMPLATE, _DEDUP_SYSTEM
    
    st.markdown("**1. Finding Public Comps (sent to Chorus models)**")
    st.code(_COMPS_QUESTION_TEMPLATE, language="text")
    
    st.markdown("**2. Deduplicating Comps (sent to Summarizer model)**")
    st.code(_DEDUP_SYSTEM, language="text")
