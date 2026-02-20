# app.py
"""
Market Comps Finder — Streamlit Frontend

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Optional

import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Market Comps Finder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Imports (after path is set) ───────────────────────────────────────────────
from market_comps.comps_engine import CompsEngine
from market_comps.config import settings
from market_comps.models import CompsResult, CompanyMetrics, ScanFilters

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        padding: 2rem 2.5rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99,179,237,0.2);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .main-header h1 {
        color: #e2e8f0;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }
    .accent { color: #63b3ed; }

    /* Metric cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        color: #e2e8f0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .metric-card .sub {
        color: #64748b;
        font-size: 0.7rem;
        margin-top: 0.1rem;
    }

    /* Usage badge */
    .usage-badge {
        background: #0f2744;
        border: 1px solid #1e4a7a;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        font-size: 0.78rem;
        color: #93c5fd;
        margin-top: 0.5rem;
    }
    .usage-badge b { color: #bfdbfe; }

    /* Status badge */
    .status-ok   { color: #4ade80; font-weight: 600; }
    .status-warn { color: #fbbf24; font-weight: 600; }
    .status-err  { color: #f87171; font-weight: 600; }

    /* Table styling */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Section headers */
    .section-header {
        color: #cbd5e1;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    .sidebar-title {
        color: #63b3ed;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.5rem;
    }

    /* Description expander */
    .desc-text {
        color: #94a3b8;
        font-style: italic;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Error / info boxes */
    .info-box {
        background: #1e293b;
        border-left: 4px solid #63b3ed;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helper: formatting ────────────────────────────────────────────────────────

def fmt_currency(val: Optional[float], decimals: int = 1) -> str:
    if val is None:
        return "—"
    if val >= 1e12:
        return f"${val/1e12:.{decimals}f}T"
    if val >= 1e9:
        return f"${val/1e9:.{decimals}f}B"
    if val >= 1e6:
        return f"${val/1e6:.{decimals}f}M"
    return f"${val:,.0f}"


def fmt_multiple(val: Optional[float], suffix: str = "x") -> str:
    if val is None:
        return "—"
    return f"{val:.1f}{suffix}"


def fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val:.1f}%"


def build_dataframe(comps: list[CompanyMetrics]) -> pd.DataFrame:
    rows = []
    for m in comps:
        rows.append(
            {
                "Company": m.name,
                "Ticker": m.ticker,
                "Exchange": m.exchange or "—",
                "Country": m.country or "—",
                "Sector": m.sector or "—",
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
            }
        )
    return pd.DataFrame(rows)


def build_export_dataframe(comps: list[CompanyMetrics]) -> pd.DataFrame:
    """Raw numeric version for CSV export."""
    rows = []
    for m in comps:
        rows.append(
            {
                "Company": m.name,
                "Ticker": m.ticker,
                "Exchange": m.exchange,
                "Country": m.country,
                "Sector": m.sector,
                "Industry": m.industry,
                "Description": m.description,
                "Market_Cap_USD": m.market_cap_usd,
                "EV_USD": m.ev_usd,
                "Revenue_TTM_USD": m.revenue_ttm_usd,
                "Revenue_NTM_USD": m.revenue_ntm_usd,
                "EV_Revenue_TTM": m.ev_to_revenue_ttm,
                "EV_Revenue_NTM": m.ev_to_revenue_ntm,
                "Gross_Margin_Pct": m.gross_margin_pct,
                "EBITDA_Margin_Pct": m.ebitda_margin_pct,
                "Revenue_Growth_YoY_Pct": m.revenue_growth_yoy_pct,
                "Data_Available": m.data_available,
                "Data_Notes": m.data_notes,
            }
        )
    return pd.DataFrame(rows)


# ── Session state init ───────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state["result"] = None
if "running" not in st.session_state:
    st.session_state["running"] = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>📊 Market <span class="accent">Comps Finder</span></h1>
        <p>Find publicly traded comparable companies for any company, industry, or sub-industry.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Configuration</div>', unsafe_allow_html=True)

    api_key = st.text_input(
        "OpenRouter API Key",
        value=settings.openrouter_api_key,
        type="password",
        help="Get your key at openrouter.ai",
    )

    model_options = [
        "google/gemini-flash-1.5",
        "google/gemini-flash-1.5-8b",
        "google/gemini-2.0-flash-001",
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-3-haiku",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.3-70b-instruct",
    ]
    model = st.selectbox(
        "LLM Model",
        model_options,
        index=model_options.index(settings.default_model)
        if settings.default_model in model_options
        else 0,
    )

    st.divider()
    st.markdown('<div class="sidebar-title">🔢 Search Parameters</div>', unsafe_allow_html=True)

    n_comps = st.slider("Number of Comps", min_value=5, max_value=30, value=10, step=1)

    st.divider()
    st.markdown('<div class="sidebar-title">🔍 Filters (optional)</div>', unsafe_allow_html=True)

    filter_countries = st.text_input(
        "Countries (comma-separated)",
        "",
        placeholder="e.g. United States, Canada",
        help="Leave blank for global.",
    )
    filter_exchanges = st.text_input(
        "Exchanges (comma-separated)",
        "",
        placeholder="e.g. NYSE, NASDAQ",
    )
    filter_sectors = st.text_input(
        "Sectors (comma-separated)",
        "",
        placeholder="e.g. Technology, Healthcare",
    )
    filter_industries = st.text_input(
        "Industries (comma-separated)",
        "",
        placeholder="e.g. Cloud Software",
    )

    col_minmc, col_maxmc = st.columns(2)
    with col_minmc:
        min_mc_b = st.number_input("Min Mkt Cap ($B)", min_value=0.0, value=0.0, step=0.5)
    with col_maxmc:
        max_mc_b = st.number_input("Max Mkt Cap ($B)", min_value=0.0, value=0.0, step=10.0)

    st.divider()

    # Usage display — shown after a successful run
    result: Optional[CompsResult] = st.session_state.get("result")
    if result and result.llm_usage.call_count > 0:
        u = result.llm_usage
        st.markdown('<div class="sidebar-title">💡 LLM Usage</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="usage-badge">
                <b>Model:</b> {result.model_used}<br>
                <b>API Calls:</b> {u.call_count}<br>
                <b>Prompt tokens:</b> {u.total_prompt_tokens:,}<br>
                <b>Completion tokens:</b> {u.total_completion_tokens:,}<br>
                <b>Total tokens:</b> {u.total_tokens:,}<br>
                <b>Est. cost:</b> ${u.estimated_cost_usd:.5f}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Main input area ───────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Search",
        label_visibility="collapsed",
        placeholder="Enter a company name, industry, or sub-industry (e.g. 'Salesforce', 'cloud ERP', 'digital payments')",
        key="query_input",
    )
with col_btn:
    find_clicked = st.button(
        "🔍 Find Comps",
        type="primary",
        use_container_width=True,
        disabled=not query.strip(),
    )

# ── Run engine ────────────────────────────────────────────────────────────────
if find_clicked and query.strip():
    # Build filters
    def _split(s: str) -> list[str]:
        return [x.strip() for x in s.split(",") if x.strip()]

    filters = ScanFilters(
        countries=_split(filter_countries),
        exchanges=_split(filter_exchanges),
        sectors=_split(filter_sectors),
        industries=_split(filter_industries),
        min_market_cap_usd=min_mc_b * 1e9 if min_mc_b > 0 else None,
        max_market_cap_usd=max_mc_b * 1e9 if max_mc_b > 0 else None,
    )

    # Status strip
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.info("🔄 **Step 1 / 2** — Scanning for comparable companies via LLM…")

    try:
        engine = CompsEngine(api_key=api_key or None, model=model)
        # Monkey-patch scanner step callback for progress
        original_scan = engine._scanner.scan

        def scan_with_progress(*args, **kwargs):
            result_scan = original_scan(*args, **kwargs)
            with progress_placeholder.container():
                st.info(
                    f"✅ **Step 1 complete** — Found {len(result_scan[0])} candidates. "
                    "🔄 **Step 2 / 2** — Fetching live market metrics…"
                )
            return result_scan

        engine._scanner.scan = scan_with_progress

        with st.spinner(""):
            result = engine.run(query=query.strip(), n_comps=n_comps, filters=filters)

        st.session_state["result"] = result
        progress_placeholder.empty()

    except Exception as exc:
        progress_placeholder.empty()
        st.error(f"❌ Engine error: {exc}")
        st.session_state["result"] = None

# ── Results ───────────────────────────────────────────────────────────────────
result: Optional[CompsResult] = st.session_state.get("result")

if result is not None:
    if result.errors:
        for err in result.errors:
            st.warning(f"⚠️ {err}")

    comps = result.comps
    n_with_data = sum(1 for c in comps if c.data_available)

    # ── Summary KPIs
    st.markdown('<div class="section-header">Results Summary</div>', unsafe_allow_html=True)
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Query</div>
                <div class="value" style="font-size:1rem;">{result.query}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kc2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Comps Found</div>
                <div class="value">{len(comps)}</div>
                <div class="sub">{result.candidates_found} candidates scanned</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kc3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Data Coverage</div>
                <div class="value">{n_with_data}/{len(comps)}</div>
                <div class="sub">tickers with live data</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kc4:
        u = result.llm_usage
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">LLM Cost</div>
                <div class="value">${u.estimated_cost_usd:.4f}</div>
                <div class="sub">{u.total_tokens:,} tokens · {u.call_count} calls</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")

    if not comps:
        st.markdown(
            '<div class="info-box">No comparable companies found. Try a different query or adjust filters.</div>',
            unsafe_allow_html=True,
        )
    else:
        # ── Metrics table
        st.markdown('<div class="section-header">📋 Comparable Companies</div>', unsafe_allow_html=True)

        # Sort control
        sort_col = st.selectbox(
            "Sort by",
            ["EV/Rev (TTM)", "EV/Rev (NTM)", "Market Cap", "Gross Margin", "Rev Growth YoY", "Company"],
            index=0,
            key="sort_col",
            label_visibility="collapsed",
        )

        df = build_dataframe(comps)
        df_display = df.copy()

        # Apply sort
        sort_map = {
            "EV/Rev (TTM)": "EV/Rev (TTM)",
            "EV/Rev (NTM)": "EV/Rev (NTM)",
            "Market Cap": "Market Cap",
            "Gross Margin": "Gross Margin",
            "Rev Growth YoY": "Rev Growth YoY",
            "Company": "Company",
        }
        sort_col_actual = sort_map[sort_col]

        # Sort numeric-ish: strip $, B, x etc.
        def _sortable(val: str) -> float:
            if val in ("—", ""):
                return float("-inf")
            s = val.replace("$", "").replace(",", "").replace("x", "").replace("%", "").strip()
            try:
                mult = 1
                if s.endswith("T"):
                    mult = 1e12
                    s = s[:-1]
                elif s.endswith("B"):
                    mult = 1e9
                    s = s[:-1]
                elif s.endswith("M"):
                    mult = 1e6
                    s = s[:-1]
                return float(s) * mult
            except ValueError:
                return float("-inf")

        if sort_col_actual != "Company":
            sort_vals = df_display[sort_col_actual].apply(_sortable)
            df_display = df_display.iloc[sort_vals.argsort()[::-1].values]

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Data": st.column_config.TextColumn("Data", width=60),
                "Ticker": st.column_config.TextColumn("Ticker", width=90),
                "Exchange": st.column_config.TextColumn("Exchange", width=90),
                "Country": st.column_config.TextColumn("Country", width=100),
            },
        )

        # ── Company detail expanders
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
                    st.markdown(
                        f'<div class="desc-text">{m.description}</div>',
                        unsafe_allow_html=True,
                    )
                if not m.data_available:
                    st.warning(f"⚠️ Data unavailable: {m.data_notes}")

        # ── Export
        st.markdown('<div class="section-header">⬇️ Export</div>', unsafe_allow_html=True)
        export_df = build_export_dataframe(comps)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        safe_query = "".join(c if c.isalnum() else "_" for c in result.query)[:40]
        st.download_button(
            "📥 Download CSV",
            data=csv_bytes,
            file_name=f"market_comps_{safe_query}.csv",
            mime="text/csv",
            type="secondary",
        )

elif result is None and not find_clicked:
    # Initial empty state
    st.markdown(
        """
        <div class="info-box">
            👆 Enter a <b>company name</b>, <b>industry</b>, or <b>sub-industry</b> above and click <b>Find Comps</b>.
            <br><br>
            Examples: <i>Salesforce</i> · <i>cloud ERP</i> · <i>digital payments</i> · <i>semiconductor equipment</i> · <i>specialty pharma</i>
        </div>
        """,
        unsafe_allow_html=True,
    )
