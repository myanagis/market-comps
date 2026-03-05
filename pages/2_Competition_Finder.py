# pages/2_Competition_Finder.py
"""
Competition Finder — Discover competitors (public & private) across adjacent markets.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Competition Finder",
    page_icon="🏢",
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
from market_comps.competition.competition_finder import CompetitionFinder, CompetitionResult, Competitor
from market_comps.config import settings, MODEL_OPTIONS
from market_comps.ui import inject_global_style, create_chorus_progress_status
from market_comps.cross_checker.cross_checker import DEFAULT_MODELS as _CHORUS_DEFAULTS

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

.public-badge  { background: #0d3b2e; border: 1px solid #16a34a; color: #4ade80; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.private-badge { background: #2d1b4e; border: 1px solid #7c3aed; color: #a78bfa; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
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

def fmt_int(val: Optional[int]) -> str:
    return "—" if val is None else str(val)

def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        h = urlparse(url).netloc
        return h.replace("www.", "") or url[:40]
    except Exception:
        return url[:40]

def build_competition_data(result: CompetitionResult) -> tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    url_to_id = {}
    footnotes = []
    
    def _get_footnote_markers(urls: list[str], html: bool = False) -> str:
        if not urls: return ""
        markers = []
        for u in urls:
            if u not in url_to_id:
                url_to_id[u] = len(url_to_id) + 1
                footnotes.append((url_to_id[u], u))
            if html:
                markers.append(f'<a href="{u}" class="fn-link" target="_blank" title="{u}">[{url_to_id[u]}]</a>')
            else:
                markers.append(f"[{url_to_id[u]}]")
        return "".join(markers)

    pub_rows_df, priv_rows_df = [], []
    pub_rows_html, priv_rows_html = [], []
    
    for c in result.competitors:
        markers_txt = _get_footnote_markers(c.source_urls, html=False)
        markers_html = _get_footnote_markers(c.source_urls, html=True)
        
        name_txt = f"{c.name} {markers_txt}".strip()
        name_html = f"<strong>{c.name}</strong>{markers_html}"
        
        if c.type == "public":
            row_dict = {
                "Company": name_txt,
                "Ticker": c.ticker or "—",
                "Country": getattr(c, "country", None) or "—",
                "Market Cap": fmt_currency(c.market_cap_usd),
                "Revenue": fmt_currency(c.revenue_usd),
                "EV/Revenue": fmt_multiple(c.ev_to_revenue)
            }
            pub_rows_df.append(row_dict)
            pub_rows_html.append([name_html, row_dict["Ticker"], row_dict["Country"], row_dict["Market Cap"], row_dict["Revenue"], row_dict["EV/Revenue"]])
        else:
            inv = ", ".join(getattr(c, "investors", [])) or "—"
            exit_acq = getattr(c, "exit_acquirer", None)
            exit_amt = fmt_currency(getattr(c, "exit_amount_usd", None))
            exit_d = getattr(c, "exit_date", None)
            if exit_acq:
                exit_str = f"Acq. by {exit_acq}"
                if exit_amt != "—": exit_str += f" ({exit_amt})"
                if exit_d: exit_str += f" in {exit_d}"
            else:
                exit_str = "—"
                
            row_dict = {
                "Company": name_txt,
                "Country": getattr(c, "country", None) or "—",
                "Latest Round": c.latest_round or "—",
                "Amount Raised": fmt_currency(c.amount_raised_usd),
                "Investors": inv,
                "Exit": exit_str
            }
            priv_rows_df.append(row_dict)
            priv_rows_html.append([name_html, row_dict["Country"], row_dict["Latest Round"], row_dict["Amount Raised"], row_dict["Investors"], row_dict["Exit"]])
            
    def _tbl(headers, rows):
        if not rows: return "<p><em>No data.</em></p>"
        h = "".join(f"<th>{x}</th>" for x in headers)
        r = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
        return f'<table class="printable-table"><thead><tr>{h}</tr></thead><tbody>{r}</tbody></table>'
        
    pub_html = _tbl(["Company", "Ticker", "Country", "Market Cap", "Revenue", "EV/Revenue"], pub_rows_html)
    priv_html = _tbl(["Company", "Country", "Latest Round", "Amount Raised", "Investors", "Exit"], priv_rows_html)
    
    fn_html = ""
    if footnotes:
        items = "".join(f'<p><a href="{u}" target="_blank">[{i}] {_domain(u)}</a></p>' for i, u in footnotes)
        fn_html = f'<div class="legend-box"><strong>Sources</strong>{items}</div>'
        
    return pd.DataFrame(pub_rows_df), pd.DataFrame(priv_rows_df), fn_html, pub_html, priv_html

def generate_html_report(query: str, comp_result: CompetitionResult) -> str:
    from datetime import datetime
    now = datetime.now().strftime("%B %d, %Y %H:%M")

    comp_html = ""
    if comp_result:
        landscape = comp_result.landscape or ""
        _, _, fn_tHtml, pub_tHtml, priv_tHtml = build_competition_data(comp_result)
        details = ""
        for c in comp_result.competitors:
            sources = " ".join(f'<a href="{u}" target="_blank">{_domain(u)}</a>' for u in c.source_urls)
            badge = "PUBLIC" if c.type == "public" else "PRIVATE"
            if c.type == "public":
                fin = f"Market Cap: {fmt_currency(c.market_cap_usd)} &nbsp;|&nbsp; Revenue: {fmt_currency(c.revenue_usd)} &nbsp;|&nbsp; EV/Rev: {fmt_multiple(c.ev_to_revenue)}"
            else:
                inv = ", ".join(getattr(c, "investors", [])) or "—"
                exit_acq = getattr(c, "exit_acquirer", None)
                exit_amt = fmt_currency(getattr(c, "exit_amount_usd", None))
                exit_d = getattr(c, "exit_date", None)
                if exit_acq:
                    exit_str = f"Acq. by {exit_acq}"
                    if exit_amt != "—": exit_str += f" ({exit_amt})"
                    if exit_d: exit_str += f" in {exit_d}"
                else:
                    exit_str = "—"
                fin = f"Latest Round: {c.latest_round or '—'} &nbsp;|&nbsp; Raised: {fmt_currency(c.amount_raised_usd)} &nbsp;|&nbsp; Year: {fmt_int(c.funding_year)} &nbsp;|&nbsp; Investors: {inv} &nbsp;|&nbsp; Exit: {exit_str}"
            details += f"""
            <div class="card">
              <div class="card-header"><span class="badge badge-{c.type}">{badge}</span> <strong>{c.name}</strong>{' (' + c.ticker + ')' if c.ticker else ''}{(' &mdash; ' + getattr(c, 'country', None)) if getattr(c, 'country', None) else ''}</div>
              <p>{c.description}</p>
              <p><em>Differentiator:</em> {c.differentiation}</p>
              <p class="fin">{fin}</p>
              {('<p class="sources">Sources: ' + sources + '</p>') if sources else ''}
            </div>"""
        
        comp_html = f"""
        <h2>Competitive Landscape</h2>
        <p>{landscape}</p>
        <h3>Public Competitors</h3>{pub_tHtml}
        <h3>Private Competitors</h3>{priv_tHtml}
        {fn_tHtml}
        <h3>Competitor Details</h3>{details}
        """

    css = """
    body{font-family:'Segoe UI',Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#1e293b;background:#fff}
    h1{color:#1e40af;border-bottom:2px solid #3b82f6;padding-bottom:8px}
    h2{color:#1e293b;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:2rem}
    h3{color:#475569;margin-top:1.5rem}
    table{border-collapse:collapse;width:100%;margin:12px 0;font-size:0.88rem}
    th{background:#1e40af;color:#fff;padding:8px 10px;text-align:left}
    td{padding:7px 10px;border-bottom:1px solid #e2e8f0}
    tr:nth-child(even) td{background:#f8fafc}
    .card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px 18px;margin-bottom:12px}
    .card-header{font-size:1rem;margin-bottom:6px}
    .badge{padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700}
    .badge-public{background:#dcfce7;color:#166534}
    .badge-private{background:#ede9fe;color:#5b21b6}
    .fin{color:#475569;font-size:0.88rem}
    .sources{font-size:0.82rem;color:#64748b}
    .sources a{color:#2563eb}
    .meta{color:#94a3b8;font-size:0.82rem;margin-bottom:2rem}
    """
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
    <title>Competition Finder: {query}</title>
    <style>{css}</style></head><body>
    <h1>Competition Report</h1>
    <p class="meta">Query: <strong>{query}</strong> &nbsp;&mdash;&nbsp; Generated: {now}</p>
    {comp_html}
    </body></html>"""


# ── Session state ─────────────────────────────────────────────────────────────
if "comp_result" not in st.session_state:
    st.session_state["comp_result"] = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏢 <span class="accent">Competition</span> <span style="color:#64748b">Finder</span></h1>
    <p>Discover the best competitors (public &amp; private) utilizing a two-pass LLM pipeline.</p>
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

def format_model(m: str) -> str:
    in_price, out_price = settings.get_model_pricing(m)
    return f"{m} (${in_price:.2f} / ${out_price:.2f})"

with st.expander("⚙️ Advanced Options", expanded=False):
    _ao1, _ao2 = st.columns([3, 1])
    with _ao1:
        chorus_models = st.multiselect(
            "Chorus models (queried in parallel for public & private comps)",
            options=MODEL_OPTIONS,
            default=_CHORUS_DEFAULTS,
            format_func=format_model,
        )
    with _ao2:
        n_comps = st.slider("Number of Comps", min_value=5, max_value=30, value=15, step=1)

# ── Single action button ──────────────────────────────────────────────────────
_b1, _bpad = st.columns([2, 6])
with _b1:
    run_clicked = st.button(
        "🔍 Find Competitors",
        type="primary",
        disabled=not query.strip() or not chorus_models,
        use_container_width=True,
    )

if st.session_state.get("comp_result") is None and not run_clicked:
    st.markdown("""
    <div class="info-box">
        👆 Enter a <b>company name</b> or <b>industry</b> and click <b>Find Competitors</b>.<br><br>
        1. <b>Sourcing:</b> Query 5 models in parallel to source up to 30 candidates.<br>
        2. <b>Filtering:</b> Extracts the best matching comps (public and private).
    </div>
    """, unsafe_allow_html=True)

# ── Run on button click ────────────────────────────────────────
if run_clicked and query.strip() and chorus_models:
    st.session_state["comp_result"] = None

    status_comp, on_done_comp = create_chorus_progress_status(
        status_label=f"⏳ **Phase 1/2** — Querying {len(chorus_models)} models for competitors of **{query}**…",
        models=chorus_models,
    )
    with status_comp:
        try:
            finder = CompetitionFinder(chorus_models=chorus_models)
            comp_result = finder.run(
                company=query.strip(),
                description=description.strip(),
                competitors_to_include=competitors_to_include.strip(),
                limit=n_comps,
                on_model_complete=on_done_comp,
            )
            st.session_state["comp_result"] = comp_result
            status_comp.update(label=f"✅ **Phase 1/2** — Competition done", state="complete", expanded=False)
        except Exception as exc:
            status_comp.update(label=f"❌ Competition Finder error: {exc}", state="error")
            st.error(f"❌ Competition Finder error: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# Results: Competition
# ══════════════════════════════════════════════════════════════════════════════
comp_result: Optional[CompetitionResult] = st.session_state.get("comp_result")

if comp_result is not None:
    st.markdown('<div class="section-header">🏢 Competitive Landscape</div>', unsafe_allow_html=True)

    if comp_result.errors:
        for e in comp_result.errors:
            st.warning(f"⚠️ {e}")

    if comp_result.landscape:
        st.markdown(comp_result.landscape)

    pub_df, priv_df, fn_html, _, _ = build_competition_data(comp_result)

    cu = comp_result.total_llm_usage
    st.markdown(
        f'<div class="usage-badge"><b>LLM usage:</b> {cu.call_count} calls &nbsp;|&nbsp; '
        f'{cu.total_tokens:,} tokens &nbsp;|&nbsp; Est. ${cu.estimated_cost_usd:.5f}</div>',
        unsafe_allow_html=True,
    )

    _ct1, _ct2 = st.columns(2)

    with _ct1:
        n_pub = len(comp_result.public_competitors)
        st.markdown(f'<div class="section-header">🟢 Public Competitors <span style="font-size:0.85rem;color:#64748b">({n_pub})</span></div>', unsafe_allow_html=True)
        if pub_df.empty:
            st.caption("No public competitors identified.")
        else:
            st.dataframe(pub_df, use_container_width=True, hide_index=True)

    with _ct2:
        n_priv = len(comp_result.private_competitors)
        st.markdown(f'<div class="section-header">🟣 Private Competitors <span style="font-size:0.85rem;color:#64748b">({n_priv})</span></div>', unsafe_allow_html=True)
        if priv_df.empty:
            st.caption("No private competitors identified.")
        else:
            st.dataframe(priv_df, use_container_width=True, hide_index=True)

    if fn_html:
        st.markdown(fn_html, unsafe_allow_html=True)

    # Detailed cards
    if comp_result.competitors:
        with st.expander(f"📋 Competitor Details ({len(comp_result.competitors)} companies)", expanded=False):
            for c in comp_result.competitors:
                badge = '<span class="public-badge">PUBLIC</span>' if c.type == "public" else '<span class="private-badge">PRIVATE</span>'
                with st.expander(f"**{c.name}**" + (f" ({c.ticker})" if c.ticker else ""), expanded=False):
                    st.markdown(badge, unsafe_allow_html=True)
                    if c.description:
                        st.markdown(f"**Description:** {c.description}")
                    if c.differentiation:
                        st.markdown(f"**Differentiator:** {c.differentiation}")
                    if c.type == "public":
                        _d1, _d2, _d3 = st.columns(3)
                        _d1.metric("Market Cap", fmt_currency(c.market_cap_usd))
                        _d2.metric("Revenue", fmt_currency(c.revenue_usd))
                        _d3.metric("EV/Revenue", fmt_multiple(c.ev_to_revenue))
                    else:
                        inv = ", ".join(getattr(c, "investors", [])) or "—"
                        exit_acq = getattr(c, "exit_acquirer", None)
                        exit_amt = fmt_currency(getattr(c, "exit_amount_usd", None))
                        exit_d = getattr(c, "exit_date", None)
                        
                        _d1, _d2, _d3, _d4 = st.columns(4)
                        _d1.metric("Latest Round", c.latest_round or "—")
                        _d2.metric("Amount Raised", fmt_currency(c.amount_raised_usd))
                        _d3.metric("Year", fmt_int(c.funding_year))
                        
                        if exit_acq:
                            exit_str = f"{exit_acq}"
                            if exit_amt != "—": exit_str += f" ({exit_amt})"
                            if exit_d: exit_str += f"\n{exit_d}"
                            _d4.metric("Acquired By", exit_str)
                        else:
                            _d4.metric("Investors", inv)
                            
                    if c.source_urls:
                        links = " · ".join(f"[{_domain(u)}]({u})" for u in c.source_urls)
                        st.markdown(f"**Sources:** {links}")

    if comp_result.chorus_result:
        cr = comp_result.chorus_result
        with st.expander("🎼 Individual Model Responses (Chorus)", expanded=False):
            tab_labels = [("✅ " if r.success else "❌ ") + r.model.split("/")[-1] for r in cr.responses]
            tabs = st.tabs(tab_labels)
            for tab, resp in zip(tabs, cr.responses):
                with tab:
                    st.caption(f"`{resp.model}`")
                    if resp.success:
                        st.markdown(resp.content)
                    else:
                        st.error(resp.error)

    st.markdown('<div class="section-header">⬇️ Download Report</div>', unsafe_allow_html=True)
    _report_query = query.strip() or comp_result.company
    html_bytes = generate_html_report(_report_query, comp_result).encode("utf-8")
    safe_q = "".join(c if c.isalnum() else "_" for c in _report_query)[:40]
    st.download_button(
        "📄 Download HTML Report",
        data=html_bytes,
        file_name=f"competition_report_{safe_q}.html",
        mime="text/html",
        type="primary"
    )

with st.expander("🤖 How this page works", expanded=False):
    st.markdown("""
<div class="how-it-works">
<h4>Two-Pass Sourcing + Filtering</h4>
1. <b>Sourcing:</b> Query 5 models in parallel to source up to 30 candidate competitors.<br>
2. <b>Filtering:</b> Feed aggregated candidates to a Summary LLM to extract the {n} best comps.
</div>
""", unsafe_allow_html=True)
    from market_comps.competition.competition_finder import _COMPETITION_QUESTION_TEMPLATE, _EXTRACTION_SYSTEM_TEMPLATE
    st.markdown("**1. Finding Candidates**")
    st.code(_COMPETITION_QUESTION_TEMPLATE, language="text")
    st.markdown("**2. Filtering Top Competitors**")
    st.code(_EXTRACTION_SYSTEM_TEMPLATE, language="text")
