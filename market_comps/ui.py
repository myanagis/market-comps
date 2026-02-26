"""
market_comps/ui.py
Shared Streamlit UI components.
"""
from __future__ import annotations

from typing import Callable
import streamlit as st


def create_chorus_progress_status(
    status_label: str,
    models: list[str],
) -> tuple[object, Callable]:
    """
    Creates an st.status container and returns an on_model_complete callback.
    
    Usage:
        import streamlit as st
        from market_comps.ui import create_chorus_progress_status
        
        status, on_done = create_chorus_progress_status("Querying models...", ["model1", "model2"])
        with status:
            result = chorus.run(..., on_model_complete=on_done)
            status.update(label="Done!", state="complete", expanded=False)
    """
    n_models = len(models)
    completed_count = [0]
    model_lines: dict[str, object] = {}

    status = st.status(status_label, expanded=True)
    
    with status:
        # Pre-create empty placeholders so lines appear in exact model order
        for m in models:
            model_lines[m] = st.empty()
            model_lines[m].markdown(f"🕐 `{m.split('/')[-1]}` — waiting…")

    def _on_model_complete(resp):
        completed_count[0] += 1
        icon = "✅" if resp.success else "❌"
        t = f"{resp.elapsed_seconds:.1f}s"
        short = resp.model.split("/")[-1]
        
        # In case the model wasn't in the initial list (shouldn't happen, but safe fallback)
        if resp.model not in model_lines:
            with status:
                model_lines[resp.model] = st.empty()
                
        if resp.success:
            model_lines[resp.model].markdown(f"{icon} `{short}` — done in {t}")
        else:
            model_lines[resp.model].markdown(f"{icon} `{short}` — error ({t})")
            
        status.update(label=f"⏳ {completed_count[0]}/{n_models} models complete…")

    return status, _on_model_complete

def inject_global_style() -> None:
    """Injects the global CSS derived from the print HTML report."""
    st.markdown("""
    <style>
    /* Base typography & theme overrides (using !important to override Streamlit defaults) */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Segoe UI', Arial, sans-serif !important; 
    }
    
    /* Headers */
    h1 { color: #1e40af !important; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }
    h2 { color: #1e293b !important; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-top: 2rem; }
    h3 { color: #475569 !important; margin-top: 1.5rem; }

    /* Tables (for raw HTML tables) */
    .printable-table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.88rem; }
    .printable-table th { background: #1e40af; color: #fff; padding: 8px 10px; text-align: left; }
    .printable-table td { padding: 7px 10px; border-bottom: 1px solid #e2e8f0; color: #1e293b; }
    .printable-table tr:nth-child(even) td { background: #f8fafc; }
    
    /* Cards */
    .card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px; }
    .card-header { font-size: 1rem; margin-bottom: 6px; color: #1e293b; }
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .badge-public { background: #dcfce7; color: #166534; }
    .badge-private { background: #ede9fe; color: #5b21b6; }
    .fin { color: #475569; font-size: 0.88rem; }
    .sources { font-size: 0.82rem; color: #64748b; }
    .sources a { color: #2563eb; text-decoration: none; }
    .sources a:hover { text-decoration: underline; }
    .meta { color: #94a3b8; font-size: 0.82rem; margin-bottom: 2rem; }
    
    /* Footnotes */
    .fn-link { font-size: 0.75rem; vertical-align: super; color: #3b82f6; text-decoration: none; margin-left: 2px; }
    .fn-link:hover { text-decoration: underline; }
    .legend-box { font-size: 0.8rem; background: #f1f5f9; border-radius: 6px; padding: 12px; margin-top: 1rem; border: 1px solid #e2e8f0; }
    .legend-box p { margin: 2px 0; color: #475569; }
    .legend-box a { color: #2563eb; text-decoration: none; }
    .legend-box a:hover { text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True)
