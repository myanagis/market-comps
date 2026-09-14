import streamlit as st

def inject_global_styles():
    st.markdown(
        """
        <style>
        /* Typography System */
        /* Make headings consistent and slightly tighter */
        h1, .stHeadingContainer h1 {
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }
        
        /* H2 major sections */
        h2, .stHeadingContainer h2 {
            margin-top: 48px !important;
            margin-bottom: 16px !important;
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            color: #ffffff !important;
            background-color: #1e3a8a !important; /* Dark blue banner */
            padding: 8px 16px !important;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* H3 subsections */
        h3, .stHeadingContainer h3 {
            margin-top: 32px !important;
            margin-bottom: 12px !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            color: #1e40af !important; /* Slightly lighter blue text */
            border-bottom: 2px solid #bfdbfe; /* Default fallback */
            padding-bottom: 4px;
        }
        
        /* When H3 is inside a column, apply the border to the block so it spans the full width */
        div[data-testid="stHorizontalBlock"]:has(h3) {
            border-bottom: 2px solid #bfdbfe !important;
            padding-bottom: 8px !important;
            margin-bottom: 16px !important;
            align-items: flex-end !important;
        }
        
        div[data-testid="stHorizontalBlock"]:has(h3) h3 {
            border-bottom: none !important;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        
        /* Market Map Eyebrow */
        .market-eyebrow {
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: #6b7280;
            text-transform: uppercase;
            margin-bottom: -16px;
        }
        
        /* Market Notes (Introductory text) */
        .market-notes {
            max-width: 950px;
            margin-top: 16px;
            margin-bottom: 40px;
            padding-left: 20px;
            border-left: 4px solid #e5e7eb;
            color: #374151;
            line-height: 1.6;
            font-size: 1rem;
        }
        
        .market-notes p {
            margin-bottom: 16px;
        }
        
        /* Spacing for popovers aligned right next to headers */
        .header-action-container {
            display: flex;
            justify-content: flex-end;
            align-items: flex-end;
            height: 100%;
            padding-bottom: 8px; /* Align with border-bottom of H2/H3 */
        }
        
        /* -------------------------------------
           Tables via Streamlit Columns styling
           ------------------------------------- */
           
        /* Header Rows (Identify via strong tags in columns) */
        /* Streamlit nests deeply, so we need a broader :has selector */
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMarkdownContainer"] p strong) {
            background-color: #f8f9fa !important;
            padding-top: 6px !important;
            padding-bottom: 6px !important;
            padding-left: 8px !important;
            padding-right: 8px !important;
            border-bottom: 2px solid #e5e7eb !important;
            border-top: 1px solid #f3f4f6 !important;
            margin-bottom: 2px !important;
            border-radius: 4px 4px 0 0;
            gap: 0.5rem !important;
        }
        
        /* Make column header text slightly muted but bold */
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMarkdownContainer"] p strong) p {
            color: #4b5563 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            margin-bottom: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Data Rows (Exclude headers, target generic column blocks that don't have buttons with strong) */
        div[data-testid="stHorizontalBlock"]:not(:has(div[data-testid="stMarkdownContainer"] p strong)) {
            padding-top: 4px !important;
            padding-bottom: 4px !important;
            padding-left: 8px !important;
            padding-right: 8px !important;
            border-bottom: 1px solid #f3f4f6 !important;
            transition: background-color 0.15s ease-in-out;
            gap: 0.5rem !important;
        }
        
        /* Remove internal paragraph margins in data rows to eliminate whitespace */
        div[data-testid="stHorizontalBlock"]:not(:has(h1, h2, h3)) div[data-testid="stMarkdownContainer"] p {
            margin-bottom: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Data Row Hover State */
        div[data-testid="stHorizontalBlock"]:not(:has(div[data-testid="stMarkdownContainer"] p strong)):hover {
            background-color: #f9fafb !important;
        }
        
        /* Exclude form rows or columns used purely for layout from the hover effect if needed, 
           but Streamlit uses nested blocks. We only want root-level list blocks. 
           This CSS is aggressive, but works for the current table emulation strategy. */
           
        /* Action Popover Buttons in Tables */
        /* Target buttons inside popovers inside columns */
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            border: none !important;
            background: transparent !important;
            color: #9ca3af !important;
            font-size: 0.75rem !important;
            padding: 0px 4px !important;
            min-height: 0 !important;
            height: auto !important;
        }
        
        div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
            color: #374151 !important;
            background: #f3f4f6 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
