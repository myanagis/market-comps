# pages/5_FAQ.py
"""
FAQ Page — Answers to common questions, including data privacy and model training.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="FAQ",
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from market_comps.ui import inject_global_style

inject_global_style()

st.markdown("""
<style>
.faq-container { max-width: 800px; margin: 0 auto; padding-top: 1rem; }
.faq-q { color: #e2e8f0; font-size: 1.15rem; font-weight: 600; margin: 1.5rem 0 0.5rem 0; }
.faq-a { color: #94a3b8; font-size: 1rem; line-height: 1.6; margin-bottom: 1.5rem; padding-left: 1rem; border-left: 2px solid #334155; }
.faq-a a { color: #818cf8; text-decoration: none; }
.faq-a a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1>❓ FAQ</h1>
<p>Common questions about Market Comps, the Chorus of LLMs, and data privacy.</p>

<div class="faq-container">

<div class="faq-q">Is my data used to train the AI models?</div>
<div class="faq-a">
No. Market Comps uses the <b><a href="https://openrouter.ai/" target="_blank">OpenRouter API</a></b> to route queries to various foundational models (like OpenAI, Anthropic, Google, Meta, etc.). 
<br><br>
Because we use the API, your inputs and documents are <b>not used for public model training</b> by default. However, precise data retention policies depend on the specific LLM provider you select in the advanced options. 
<br><br>
You can review the full details and provider-specific policies on OpenRouter's documentation: <br>
<a href="https://openrouter.ai/docs/guides/privacy/logging" target="_blank">🔗 OpenRouter Privacy & Logging Policy</a>
</div>

<div class="faq-q">What is the "Chorus of LLMs"?</div>
<div class="faq-a">
Different AI models have different knowledge cutoffs, training biases, and reasoning strengths. The Chorus of LLMs runs your query through <b>multiple models simultaneously</b> (e.g. GPT-4o, Claude 3.5 Sonnet, DeepSeek V3, Llama 3). 
<br><br>
A final "summarizer" model then synthesizes all the responses, deduplicates the information, highlights where models agree or disagree, and ensures every claim is cited with the original source URL. This significantly reduces hallucinations and provides a much more robust answer than any single model could.
</div>

<div class="faq-q">Where does the financial data come from?</div>
<div class="faq-a">
When you use the <b>Public Comps</b> feature to fetch market data (like Market Cap, EV/Revenue, Gross Margin, etc.), that data is pulled in real-time from <b>Yahoo Finance</b> using the `yfinance` library. The LLMs are only used to <i>identify</i> the correct ticker symbols; the actual financial numbers are live market data, never hallucinated by an AI.
</div>

</div>
""", unsafe_allow_html=True)
