# pages/4_Company_Primer.py
"""
Company Primer — research one or more companies with the Chorus of LLMs
and extract a structured profile for each.
"""
from __future__ import annotations

import re
import sys
import logging
from typing import Optional

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Company Primer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from market_comps.company_primer import CompanyPrimerFinder, PrimerResult, CompanyProfile
from market_comps.company_primer.primer_finder import PRIMER_DEFAULT_MODELS
from market_comps.cross_checker.cross_checker import DEFAULT_MODELS as ALL_CHORUS_MODELS

from market_comps.ui import create_chorus_progress_status, inject_global_style

inject_global_style()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Company card */
.company-card { background: #1e293b; border: 1px solid #334155; border-radius: 14px;
    padding: 1.4rem 1.6rem; margin-bottom: 1.4rem; }
.company-card h2 { color: #e2e8f0; font-size: 1.2rem; font-weight: 700;
    margin: 0 0 0.2rem 0; }
.company-card .sub { color: #64748b; font-size: 0.8rem; margin-bottom: 1rem; }
.field-label { color: #94a3b8; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.6px; margin-bottom: 0.15rem; }
.field-value { color: #e2e8f0; font-size: 0.92rem; line-height: 1.6; margin-bottom: 0.8rem; }
.fact-list { color: #cbd5e1; font-size: 0.88rem; padding-left: 1.1rem;
    line-height: 1.8; margin: 0; }
.source-auth { color: #4ade80; font-size: 0.8rem; }
.source-sec  { color: #fbbf24; font-size: 0.8rem; }

.usage-badge { background: #0f2744; border: 1px solid #1e4a7a; border-radius: 8px;
    padding: 0.6rem 0.8rem; font-size: 0.78rem; color: #93c5fd; margin-top: 0.5rem; }
.usage-badge b { color: #bfdbfe; }

.info-box { background: #1e293b; border-left: 4px solid #818cf8; border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem; color: #94a3b8; font-size: 0.9rem; }

.how-it-works { color: #94a3b8; font-size: 0.88rem; line-height: 1.7; }
.how-it-works h4 { color: #cbd5e1; margin: 0.8rem 0 0.3rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Dollar-sign escape helper (prevent Streamlit LaTeX mangling) ──────────────
_DOLLAR_RE = re.compile(r'(?<![\\\$])\$(?=[\d,])')

def _md(text: str) -> None:
    st.markdown(_DOLLAR_RE.sub(r'\\$', text))

from market_comps.config import MODEL_OPTIONS as _ALL_MODEL_OPTIONS

# ── Session state ─────────────────────────────────────────────────────────────
if "primer_result" not in st.session_state:
    st.session_state["primer_result"] = None
if "primer_companies" not in st.session_state:
    st.session_state["primer_companies"] = [{"name": "", "context": ""}]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1>🏢 Company Primer</h1>
<p>Research one or more companies using a Chorus of Models to extract a definitive profile, key facts, and verified sources.</p>
""", unsafe_allow_html=True)

# ── Company input rows ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Companies to Research</div>', unsafe_allow_html=True)

companies = st.session_state["primer_companies"]

to_remove = None
for i, entry in enumerate(companies):
    c1, c2, c3 = st.columns([3, 4, 1])
    with c1:
        companies[i]["name"] = st.text_input(
            "Company name",
            value=entry["name"],
            key=f"co_name_{i}",
            placeholder="e.g. Stripe, Figma, Notion",
            label_visibility="collapsed" if i > 0 else "visible",
        )
    with c2:
        companies[i]["context"] = st.text_input(
            "URL or brief description (optional)",
            value=entry["context"],
            key=f"co_ctx_{i}",
            placeholder="e.g. https://stripe.com or 'payments API for developers'",
            label_visibility="collapsed" if i > 0 else "visible",
        )
    with c3:
        if i == 0:
            st.markdown("&nbsp;", unsafe_allow_html=True)  # align with inputs
        if len(companies) > 1:
            if st.button("✕", key=f"co_rm_{i}", help="Remove"):
                to_remove = i

if to_remove is not None:
    companies.pop(to_remove)
    st.session_state["primer_companies"] = companies
    st.rerun()

col_add, _ = st.columns([2, 8])
with col_add:
    if st.button("➕ Add company", type="secondary", use_container_width=True):
        companies.append({"name": "", "context": ""})
        st.session_state["primer_companies"] = companies
        st.rerun()

# ── Config ────────────────────────────────────────────────────────────────────
def format_model(m: str) -> str:
    from market_comps.config import settings
    in_price, out_price = settings.get_model_pricing(m)
    return f"{m} (${in_price:.2f} / ${out_price:.2f})"

with st.expander("⚙️ Chorus Configuration", expanded=False):
    selected_models = st.multiselect(
        "Models (queried in parallel per company)",
        options=_ALL_MODEL_OPTIONS,
        default=PRIMER_DEFAULT_MODELS,
        format_func=format_model,
        help="All selected models are queried simultaneously for each company. Prices shown: $input / $output per 1M tokens.",
    )

valid_companies = [c for c in companies if c["name"].strip()]
can_run = bool(valid_companies) and bool(
    "selected_models" in dir() and selected_models
    if "selected_models" in dir() else PRIMER_DEFAULT_MODELS
)

# Ensure selected_models is always defined
if "selected_models" not in dir():
    selected_models = PRIMER_DEFAULT_MODELS

# ── Run button ────────────────────────────────────────────────────────────────
_rb1, _ = st.columns([3, 7])
with _rb1:
    run_clicked = st.button(
        "🔍 Research Companies",
        type="primary",
        disabled=not valid_companies or not selected_models,
        use_container_width=True,
    )

if not valid_companies and not st.session_state.get("primer_result"):
    st.markdown("""
    <div class="info-box">
        👆 Add at least one company name above, then click <b>Research Companies</b>.<br><br>
        Optionally include a URL or brief description to help the models focus.
    </div>
    """, unsafe_allow_html=True)

# ── Run ───────────────────────────────────────────────────────────────────────
if run_clicked and valid_companies and selected_models:
    st.session_state["primer_result"] = None
    finder = CompanyPrimerFinder(chorus_models=selected_models)

    all_profiles: list[CompanyProfile] = []
    result_placeholder = st.empty()

    for entry in valid_companies:
        name = entry["name"].strip()
        context = entry["context"].strip()

        status, on_done = create_chorus_progress_status(
            status_label=f"🏢 Researching **{name}**…",
            models=selected_models,
        )

        with status:
            try:
                profile = finder.run_one(
                    name=name,
                    context=context,
                    on_model_complete=on_done,
                )
                all_profiles.append(profile)
                t = profile.total_usage
                lbl = f"✅ **{name}** — done · {t.total_tokens:,} tokens · ${t.estimated_cost_usd:.5f}"
                if profile.error:
                    lbl = f"⚠️ **{name}** — partial ({profile.error})"
                status.update(label=lbl, state="complete", expanded=False)
            except Exception as exc:
                status.update(label=f"❌ **{name}** — {exc}", state="error")

    st.session_state["primer_result"] = PrimerResult(profiles=all_profiles)


# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════
result: Optional[PrimerResult] = st.session_state.get("primer_result")

if result and result.profiles:
    st.markdown('<div class="section-header">📋 Company Profiles</div>', unsafe_allow_html=True)

    for profile in result.profiles:
        err_note = f" ⚠️ _{profile.error}_" if profile.error else ""
        st.markdown(f"""
        <div class="company-card">
            <h2>{profile.name}</h2>
            <div class="sub">{profile.context or "&nbsp;"}{err_note}</div>
        """, unsafe_allow_html=True)

        # 4 field columns across the top
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.markdown('<div class="field-label">Industry</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-value">{profile.industry or "—"}</div>', unsafe_allow_html=True)
        with f2:
            st.markdown('<div class="field-label">Target Customer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-value">{profile.target_customer or "—"}</div>', unsafe_allow_html=True)
        with f3:
            st.markdown('<div class="field-label">HQ Location</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-value">{profile.location or "—"}</div>', unsafe_allow_html=True)
        with f4:
            t = profile.total_usage
            st.markdown('<div class="field-label">LLM Usage</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="field-value" style="font-size:0.8rem;">'
                f'{t.call_count} calls · {t.total_tokens:,} tokens<br>'
                f'Est. ${t.estimated_cost_usd:.5f}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # Description
        if profile.description:
            st.markdown('<div class="field-label" style="margin-top:0.5rem;">Description</div>', unsafe_allow_html=True)
            _md(profile.description)

        # Key facts
        if profile.key_facts:
            st.markdown('<div class="field-label" style="margin-top:0.6rem;">Key Facts</div>', unsafe_allow_html=True)
            for fact in profile.key_facts:
                st.markdown(f"- {fact}")

        # Sources
        if profile.sources:
            with st.expander("📎 Sources", expanded=False):
                for src in profile.sources:
                    tier_icon = "✅" if src.tier == "authoritative" else "⚠️"
                    tier_label = "Authoritative" if src.tier == "authoritative" else "Secondary"
                    label = src.label or src.url
                    st.markdown(f"{tier_icon} [{label}]({src.url}) — *{tier_label}*")

        st.markdown("---")

    # ── Total usage summary ───────────────────────────────────────────────────
    total = result.total_usage
    st.markdown(
        f'<div class="usage-badge">'
        f'<b>Total LLM usage:</b> {total.call_count} calls &nbsp;|&nbsp; '
        f'{total.total_tokens:,} tokens &nbsp;|&nbsp; '
        f'Est. ${total.estimated_cost_usd:.5f}</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# How It Works
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🤖 How this page works", expanded=False):
    st.markdown("""
<div class="how-it-works">

<h4>Step 1 — Chorus Research</h4>
For each company, the selected LLMs are queried in parallel with a structured research question.
The question asks for description, industry, target customer, location, and key facts — with strict
citation requirements.

<h4>Step 2 — Structured Extraction</h4>
A summarizer model reads all model responses and reconciles them into a single clean JSON profile.
Conflicting facts are surfaced; unknown fields are left blank rather than guessed.

<h4>Source Reliability</h4>
Every cited URL is classified as:
- ✅ <b>Authoritative</b>: official company site, government sources, Wikipedia, major news orgs
- ⚠️ <b>Secondary</b>: blogs, newsletters, VC trackers, niche industry sites

<h4>Anti-hallucination rules</h4>
Models are instructed to only state facts they are confident about, cite every claim,
and explicitly say "unknown" rather than guess. The extraction step also deduplicates
and rejects facts not present in the raw research.

</div>
""", unsafe_allow_html=True)

    st.markdown("#### LLM Prompts Used")
    from market_comps.company_primer.primer_finder import _RESEARCH_QUESTION_TEMPLATE, _EXTRACTION_SYSTEM
    
    st.markdown("**1. Initial Research (sent to Chorus models)**")
    st.code(_RESEARCH_QUESTION_TEMPLATE, language="text")
    
    st.markdown("**2. Synthesis (sent to Summarizer model)**")
    st.code(_EXTRACTION_SYSTEM, language="text")
