# pages/3_LLM_Cross_Checker.py
"""
LLM Cross Checker — ask a question, get answers from multiple models, read the synthesis.
"""
from __future__ import annotations

import re

import streamlit as st

from market_comps.config import settings
from market_comps.cross_checker import LLMChorus, ChorusResult
from market_comps.cross_checker.cross_checker import (
    DEFAULT_MODELS, DEFAULT_SUMMARY_MODEL, SYSTEM_INSTRUCTIONS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DOLLAR_RE = re.compile(r'(?<![\\])\$(?=[\d,])')

def _md(text: str) -> None:
    """Render LLM text safely: escape bare $ (currency) so Streamlit's LaTeX
    renderer doesn't treat '$50M' as a LaTeX expression."""
    safe = _DOLLAR_RE.sub(r'\\$', text)
    st.markdown(safe)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chorus of LLMs",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from market_comps.ui import inject_global_style

inject_global_style()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.section-header {
    color: #cbd5e1; font-size: 1.05rem; font-weight: 600;
    border-bottom: 1px solid #334155; padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

.model-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.model-card-header {
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.6rem;
}
.model-name {
    font-size: 0.85rem; font-weight: 600;
    color: #a5b4fc; font-family: monospace;
}
.model-error {
    color: #f87171; font-size: 0.85rem; font-style: italic;
}
.model-meta {
    color: #64748b; font-size: 0.75rem; margin-top: 0.4rem;
}

.summary-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid #4f46e5;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.summary-card-header {
    color: #a5b4fc; font-size: 1.05rem; font-weight: 700; margin-bottom: 0.8rem;
}

.instructions-box {
    background: #0f172a; border-left: 4px solid #818cf8;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    font-size: 0.85rem; color: #94a3b8;
}

.stat-note { color: #64748b; font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎼 <span class="accent">Chorus of LLMs</span></h1>
    <p>Ask a question — query multiple LLMs in parallel, then synthesize a unified answer.</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "cc_result" not in st.session_state:
    st.session_state["cc_result"] = None

# ── Question input ────────────────────────────────────────────────────────────
question = st.text_area(
    "Your question",
    placeholder="e.g. What were the main causes of the 2008 financial crisis?",
    height=100,
    label_visibility="collapsed",
)

# ── System instructions (read-only) ───────────────────────────────────────────
with st.expander("📋 Model Instructions (appended automatically)", expanded=False):
    st.markdown(
        '<div class="instructions-box">'
        + SYSTEM_INSTRUCTIONS.strip().replace("\n", "<br>")
        + "</div>",
        unsafe_allow_html=True,
    )

# ── Advanced Options ──────────────────────────────────────────────────────────
# Build a broad candidate list for the multiselect
ALL_MODEL_OPTIONS = sorted(set(DEFAULT_MODELS + [
    "anthropic/claude-3-haiku",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash-preview-05-20",
    "mistralai/mistral-7b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "cohere/command-r-plus",
    "perplexity/llama-3.1-sonar-large-128k-online",
    "deepseek/deepseek-chat",
]))

with st.expander("⚙️ Advanced Options", expanded=False):
    def format_model(m: str) -> str:
        in_price, out_price = settings.get_model_pricing(m)
        return f"{m} (${in_price:.2f} / ${out_price:.2f})"

    _ao1, _ao2 = st.columns([3, 1])
    with _ao1:
        selected_models = st.multiselect(
            "Models to query",
            options=ALL_MODEL_OPTIONS,
            default=DEFAULT_MODELS,
            format_func=format_model,
            help="All selected models are queried in parallel.",
        )
    with _ao2:
        summary_model = st.selectbox(
            "Summary model",
            options=[DEFAULT_SUMMARY_MODEL] + [m for m in ALL_MODEL_OPTIONS if m != DEFAULT_SUMMARY_MODEL],
            index=0,
            format_func=format_model,
            help="Model used to synthesize a unified answer from all responses.",
        )

from market_comps.ui import create_chorus_progress_status

# ── Run ───────────────────────────────────────────────────────────────────────
run_clicked = st.button(
    "🔍 Generate Response",
    type="primary",
    disabled=not question.strip() or not selected_models,
)

if run_clicked and question.strip() and selected_models:
    st.session_state["cc_result"] = None
    checker = LLMChorus()

    status, on_done = create_chorus_progress_status(
        status_label=f"⏳ Querying {len(selected_models)} models…",
        models=selected_models,
    )

    with status:
        try:
            result = checker.run(
                question=question.strip(),
                models=selected_models,
                summary_model=summary_model,
                on_model_complete=on_done,
            )
            status.update(label=f"🧠 Synthesizing with `{summary_model}`…")
            st.session_state["cc_result"] = result
            n_ok = sum(1 for r in result.responses if r.success)
            status.update(
                label=f"✅ Done — {n_ok}/{n_models} models · {result.total_elapsed_seconds:.1f}s",
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            status.update(label=f"❌ Error: {exc}", state="error")
            st.error(f"❌ Unexpected error: {exc}")

# ── Results ───────────────────────────────────────────────────────────────────
result: ChorusResult | None = st.session_state.get("cc_result")

if result is not None:

    # ── 1. Synthesized summary (shown first) ──────────────────────────────
    if result.summary or result.summary_error:
        st.markdown('<div class="section-header">� Synthesized Answer</div>', unsafe_allow_html=True)

        if result.summary_error:
            st.error(f"Summarization failed: {result.summary_error}")
        else:
            st.markdown(
                '<div class="summary-card">'
                '<div class="summary-card-header">🧠 Unified Answer</div>',
                unsafe_allow_html=True,
            )
            _md(result.summary)
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption(
                f"Summary model: `{result.summary_model}` &nbsp;|&nbsp; "
                f"Tokens: {result.summary_usage.total_tokens:,} &nbsp;|&nbsp; "
                f"Cost: ${result.summary_usage.estimated_cost_usd:.5f}"
            )

    # ── 2. Per-model responses in tabs (collapsible) ──────────────────────
    with st.expander("🤖 Individual Model Responses", expanded=False):
        tab_labels = []
        for resp in result.responses:
            short = resp.model.split("/")[-1]
            icon = "✅" if resp.success else "❌"
            elapsed = getattr(resp, "elapsed_seconds", None)
            label = f"{icon} {short}"
            if elapsed is not None:
                label += f" ({elapsed:.1f}s)"
            tab_labels.append(label)

        tabs = st.tabs(tab_labels)
        for tab, resp in zip(tabs, result.responses):
            with tab:
                st.caption(f"`{resp.model}`")
                if resp.success:
                    _md(resp.content)
                    elapsed = getattr(resp, "elapsed_seconds", None)
                    meta = (
                        f"Tokens: {resp.usage.total_tokens:,} &nbsp;|&nbsp; "
                        f"Cost: ${resp.usage.estimated_cost_usd:.5f}"
                    )
                    if elapsed is not None:
                        meta += f" &nbsp;|&nbsp; Time: {elapsed:.1f}s"
                    st.markdown(f'<div class="model-meta">{meta}</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error: {resp.error}")


    # Pre-compute totals (needed for expander label + metric cards)
    import pandas as pd
    t = result.total_usage
    total_elapsed = getattr(result, "total_elapsed_seconds", None)

    # ── 2 & 3. Model responses + usage stats (collapsible) ────────────────
    expander_label = (
        f"🤖 Individual Model Responses & Usage Stats "
        f"— {sum(1 for r in result.responses if r.success)}/{len(result.responses)} OK"
        + (f", {total_elapsed:.1f}s" if total_elapsed else "")
        + f", ${t.estimated_cost_usd:.5f}"
    )
    with st.expander(expander_label, expanded=False):

        # Tabs
        tab_labels = []
        for resp in result.responses:
            short = resp.model.split("/")[-1]
            icon = "✅" if resp.success else "❌"
            elapsed = getattr(resp, "elapsed_seconds", None)
            label = f"{icon} {short}"
            if elapsed is not None:
                label += f" ({elapsed:.1f}s)"
            tab_labels.append(label)

        tabs = st.tabs(tab_labels)
        for tab, resp in zip(tabs, result.responses):
            with tab:
                st.caption(f"`{resp.model}`")
                if resp.success:
                    st.markdown(resp.content)
                    elapsed = getattr(resp, "elapsed_seconds", None)
                    meta = (
                        f"Tokens: {resp.usage.total_tokens:,} &nbsp;|&nbsp; "
                        f"Cost: ${resp.usage.estimated_cost_usd:.5f}"
                    )
                    if elapsed is not None:
                        meta += f" &nbsp;|&nbsp; Time: {elapsed:.1f}s"
                    st.markdown(f'<div class="model-meta">{meta}</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error: {resp.error}")

        st.divider()

        # Stats table
        rows = []
        for resp in result.responses:
            elapsed = getattr(resp, "elapsed_seconds", None)
            rows.append({
                "Model": resp.model,
                "Status": "✅ OK" if resp.success else "❌ Error",
                "Time (s)": f"{elapsed:.1f}" if elapsed is not None else "—",
                "Prompt tokens": resp.usage.total_prompt_tokens,
                "Completion tokens": resp.usage.total_completion_tokens,
                "Total tokens": resp.usage.total_tokens,
                "Est. Cost ($)": f"{resp.usage.estimated_cost_usd:.5f}",
            })
        rows.append({
            "Model": f"[Summary] {result.summary_model}",
            "Status": "✅ OK" if not result.summary_error else "❌ Error",
            "Time (s)": "—",
            "Prompt tokens": result.summary_usage.total_prompt_tokens,
            "Completion tokens": result.summary_usage.total_completion_tokens,
            "Total tokens": result.summary_usage.total_tokens,
            "Est. Cost ($)": f"{result.summary_usage.estimated_cost_usd:.5f}",
        })
        rows.append({
            "Model": "TOTAL",
            "Status": "",
            "Time (s)": f"{total_elapsed:.1f}" if total_elapsed is not None else "—",
            "Prompt tokens": t.total_prompt_tokens,
            "Completion tokens": t.total_completion_tokens,
            "Total tokens": t.total_tokens,
            "Est. Cost ($)": f"{t.estimated_cost_usd:.5f}",
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Metric cards
        _s1, _s2, _s3, _s4, _s5 = st.columns(5)
        _s1.metric("Models queried", len(result.responses))
        _s2.metric("Successful", sum(1 for r in result.responses if r.success))
        _s3.metric("Total tokens", f"{t.total_tokens:,}")
        _s4.metric("Total cost", f"${t.estimated_cost_usd:.5f}")
        _s5.metric("Total time", f"{total_elapsed:.1f}s" if total_elapsed else "—")


# ── How It Works ──────────────────────────────────────────────────────────────
with st.expander("🤖 How this page works", expanded=False):
    st.markdown("""
**Chorus of LLMs** queries multiple AI models in parallel and synthesizes a unified answer in two phases:

1. **Phase 1 — Parallel Queries**: Your question (plus injected citation and anti-hallucination instructions)
   is sent simultaneously to all selected models using a thread pool. Each model responds independently.

2. **Phase 2 — Synthesis**: A summarizer model reads all responses and produces a single unified answer,
   deduplicating information, preserving all cited URLs, and flagging disagreements between models.

**Source reliability tiers** are applied automatically:
- ✅ Authoritative sources (government, major news, Wikipedia, official pages)
- ⚠️ Secondary sources (niche trackers, blogs, VC databases)

Stats (per-model response time, token counts, costs) are tracked and shown in the collapsible section.
""")
    
    st.markdown("#### LLM Prompts Used")
    from market_comps.cross_checker.cross_checker import SYSTEM_INSTRUCTIONS, _SUMMARY_SYSTEM
    
    st.markdown("**1. Answering Questions (sent to Chorus models)**")
    st.markdown("Your input question is prefixed with these anti-hallucination constraints:")
    st.code(SYSTEM_INSTRUCTIONS, language="text")
    
    st.markdown("**2. Synthesis (sent to Summarizer model)**")
    st.code(_SUMMARY_SYSTEM, language="text")



