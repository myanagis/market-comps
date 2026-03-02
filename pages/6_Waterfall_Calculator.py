import streamlit as st
import pandas as pd
from datetime import date
import importlib
import json
import zlib
import base64
from streamlit_tags import st_tags

def encode_state(cap_table_data, exit_tags, exit_date):
    def default_serializer(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if hasattr(obj, 'item'):
            return obj.item()
        raise TypeError(f"Type {type(obj)} not serializable")
    state_dict = {"cap": cap_table_data, "exits": exit_tags, "date": exit_date}
    json_str = json.dumps(state_dict, default=default_serializer)
    compressed = zlib.compress(json_str.encode('utf-8'))
    return base64.urlsafe_b64encode(compressed).decode('utf-8')

def decode_state(b64_str):
    try:
        compressed = base64.urlsafe_b64decode(b64_str.encode('utf-8'))
        json_str = zlib.decompress(compressed).decode('utf-8')
        state_dict = json.loads(json_str)
        # Parse Dates (Safely handling timestamps with "T00:00:00")
        for row in state_dict.get("cap", []):
            if "date" in row and isinstance(row["date"], str): 
                row["date"] = date.fromisoformat(row["date"].split('T')[0])
            if "maturity_date" in row and isinstance(row["maturity_date"], str): 
                row["maturity_date"] = date.fromisoformat(row["maturity_date"].split('T')[0])
        if "date" in state_dict and isinstance(state_dict["date"], str):
            state_dict["date"] = date.fromisoformat(state_dict["date"].split('T')[0])
        return state_dict
    except Exception as e:
        st.error(f"Failed to load shared URL state: {e}")
        return None

import market_comps.waterfall.models
import market_comps.waterfall.calculator
importlib.reload(market_comps.waterfall.models)
importlib.reload(market_comps.waterfall.calculator)
from market_comps.waterfall.models import Security, SecurityType, WaterfallModel, ExitScenario

# Clear cache if the model definition changed, since Pydantic gets grumpy
if "models_reloaded" not in st.session_state:
    st.session_state.clear()
    st.session_state.models_reloaded = True

st.set_page_config(page_title="Waterfall Analysis", page_icon="💧", layout="wide")
st.title("💧 New Waterfall Analysis")

st.markdown("""
Input your cap table securities below. The table will automatically parse the data into our new model.
""")

# Check for shared state in URL before we setup default session state
if "share" in st.query_params and "shared_loaded" not in st.session_state:
    shared_state = decode_state(st.query_params["share"])
    if shared_state:
        st.session_state.cap_table_data = shared_state.get("cap", [])
        st.session_state.exit_tags = shared_state.get("exits", ['50.0', '100.0'])
        
        # Safely extract and parse the date
        raw_date = shared_state.get("date")
        if isinstance(raw_date, str):
            st.session_state.exit_date = date.fromisoformat(raw_date.split('T')[0])
        elif isinstance(raw_date, date):
            st.session_state.exit_date = raw_date
        else:
            st.session_state.exit_date = date(2026, 1, 1)
            
    st.session_state.shared_loaded = True

# Setup reasonable defaults for the data editor ONLY if we aren't loading a shared URL
if "cap_table_data" not in st.session_state and not st.session_state.get("shared_loaded", False):
    st.session_state.cap_table_data = [
        {
            "name": "Founders",
            "security_type": "common",
            "date": date(2023, 1, 1),
            "total_shares": 1_000_000,
            "fully_diluted_shares": 1_000_000,
            "capital_raised": 0.0,
            "liquidity_preference": 1.0,
            "interest_rate": None,
            "is_compounding": None,
            "maturity_date": None,
            "pre_money_cap": None,
            "discount": None,
            "pre_money": None,
            "post_money": None,
            "post_money_options_pool_percentage": None
        },
        {
            "name": "Seed",
            "security_type": "preferred",
            "date": date(2023, 6, 1),
            "total_shares": 250_000,
            "fully_diluted_shares": 250_000,
            "capital_raised": 2.0,
            "liquidity_preference": 1.5,
            "interest_rate": None,
            "is_compounding": None,
            "maturity_date": None,
            "pre_money_cap": None,
            "discount": None,
            "pre_money": 8.0,
            "post_money": 10.0,
            "post_money_options_pool_percentage": None
        },
        {
            "name": "Series A",
            "security_type": "convertible note",
            "date": date(2024, 1, 1),
            "total_shares": 375_000,
            "fully_diluted_shares": 375_000,
            "capital_raised": 4.5,
            "liquidity_preference": 1.0,
            "interest_rate": 10.0,
            "is_compounding": True,
            "maturity_date": date(2026, 1, 1),
            "pre_money_cap": 15.0,
            "discount": 20.0,
            "pre_money": None,
            "post_money": None,
            "post_money_options_pool_percentage": None
        }
    ]

# We need to manually sync session state and the table to avoid rendering conflicts
df = pd.DataFrame(st.session_state.cap_table_data)
    
st.write("View the high-level cap table summary below, or expand the details to edit the underlying capitalization parameters.")

summary_container = st.container()

with st.expander("Edit Cap Table Parameters", expanded=False):
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "security_type": st.column_config.SelectboxColumn(
                "Security Type",
                options=[e.value for e in SecurityType],
                required=True
            ),
            "date": st.column_config.DateColumn("Date", format="MM/DD/YYYY"),
            "total_shares": st.column_config.NumberColumn("Total Shares", format="%d"),
            "fully_diluted_shares": st.column_config.NumberColumn("FD Shares", format="%d"),
            "capital_raised": st.column_config.NumberColumn("Capital Raised ($M)", format="$%.2f"),
            "liquidity_preference": st.column_config.NumberColumn("Liq Pref (x)", format="%.2f"),
            "pre_money": st.column_config.NumberColumn("Pre-Money ($M)", format="$%.2f"),
            "post_money": st.column_config.NumberColumn("Post-Money ($M)", format="$%.2f"),
            "pre_money_cap": st.column_config.NumberColumn("Pre-Money Cap ($M)", format="$%.2f"),
        },
        use_container_width=True
    )

# Parse valid dataframe dict representation to session_state
new_records = []
for record in edited_df.to_dict('records'):
    clean_record = {k: v for k, v in record.items() if pd.notna(v)}
    new_records.append(clean_record)

st.session_state.cap_table_data = new_records

# Model Parsing
parsed_securities = []
errors = []

for idx, row_dict in enumerate(new_records):
    try:
        sec = Security(**row_dict)
        parsed_securities.append(sec)
    except Exception as e:
        errors.append(f"Row {idx + 1} ({row_dict.get('name', 'Unknown')}): {str(e)}")

if errors:
    st.error("There are validation errors in the table:")
    for err in errors:
        st.write(f"- {err}")
else:
    # Generate human-readable summary table
    summary_rows = []
    for sec in parsed_securities:
        desc_parts = []
        if sec.security_type == SecurityType.COMMON:
            if getattr(sec, 'total_shares', 0):
                desc_parts.append(f"{sec.total_shares:,} shares")
        elif sec.security_type == SecurityType.PREFERRED:
            if getattr(sec, 'capital_raised', 0):
                desc_parts.append(f"${sec.capital_raised:,.2f}M raised")
            if getattr(sec, 'pre_money', 0):
                desc_parts.append(f"${sec.pre_money:,.2f}M pre-money")
            if getattr(sec, 'liquidity_preference', 0) and getattr(sec, 'liquidity_preference', 0) != 1.0:
                desc_parts.append(f"{sec.liquidity_preference}x liq pref")
        elif sec.security_type == SecurityType.CONVERTIBLE_NOTE:
            if getattr(sec, 'capital_raised', 0):
                desc_parts.append(f"${sec.capital_raised:,.2f}M conv note")
            if getattr(sec, 'interest_rate', 0):
                interest_type = "compounding" if getattr(sec, 'is_compounding', False) else "simple"
                desc_parts.append(f"{sec.interest_rate}% {interest_type} interest")
            if getattr(sec, 'pre_money_cap', 0):
                desc_parts.append(f"${sec.pre_money_cap:,.2f}M cap")
            if getattr(sec, 'discount', 0):
                desc_parts.append(f"{sec.discount}% discount")
        elif sec.security_type == SecurityType.SAFE:
            if getattr(sec, 'capital_raised', 0):
                desc_parts.append(f"${sec.capital_raised:,.2f}M SAFE")
            if getattr(sec, 'pre_money_cap', 0):
                desc_parts.append(f"${sec.pre_money_cap:,.2f}M cap")
            if getattr(sec, 'discount', 0):
                desc_parts.append(f"{sec.discount}% discount")
                
        if not desc_parts:
            desc_parts.append(sec.security_type.value.title())
            
        summary_rows.append({
            "Date": sec.date.strftime("%m/%d/%Y"),
            "Round": sec.name,
            "Details": ", ".join(desc_parts)
        })
        
    with summary_container:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    waterfall_model = WaterfallModel(securities=parsed_securities)
    
    # Exit Scenarios
    st.divider()
    st.subheader("Exit Scenarios")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Default exits presented as pills (multiselect)
        default_exits = [50_000_000.0, 100_000_000.0, 250_000_000.0]
        
        # User can type any custom values as tag pills
        default_exits = st.session_state.get('exit_tags', ['50.0', '100.0'])
        exit_values_str = st_tags(
            label="Enter Exit Values ($M)",
            text="Press enter to add more",
            value=default_exits,
            suggestions=['10.0', '50.0', '100.0', '250.0', '500.0', '1000.0'],
            maxtags=15,
            key='exit_tags'
        )
        
        exit_values = []
        if exit_values_str:
            for val in exit_values_str:
                try:
                    # Clean the string (remove spaces, user-entered formatting commas, etc)
                    clean_val = str(val).strip().replace(',', '').replace('$', '').replace('M', '').replace('m', '')
                    exit_values.append(float(clean_val))
                except ValueError:
                    pass
        
    with col2:
        exit_date_val = st.session_state.get('exit_date', date(2026, 1, 1))
        exit_date = st.date_input("Exit Date", value=exit_date_val, format="MM/DD/YYYY", key='exit_date')
    
    st.write("")
    calc_col, share_col = st.columns([1, 1])
    
    with calc_col:
        calc_pressed = st.button("Calculate Flow of Funds", type="primary", use_container_width=True)
        
    with share_col:
        if st.button("🔗 Generate Shareable Link", use_container_width=True):
            share_code = encode_state(st.session_state.cap_table_data, st.session_state.get('exit_tags', []), st.session_state.get('exit_date', date(2026, 1, 1)))
            st.query_params["share"] = share_code
            
            # Since Streamlit doesn't have a direct clipboard API without custom JS components, 
            # we use st.code() which natively provides a quick-copy button on hover.
            # Assuming standard localhost or deployed URL, but pulling the base dynamically isn't natively supported, 
            # so we'll just present the relative / full query for them to click-to-copy.
            full_url = f"?share={share_code}"
            st.toast("Link generated! Click the copy button below.")
            st.info("Copy your shareable link here:")
            st.code(full_url, language=None)
            
    if calc_pressed:
        if not exit_values:
            st.warning("Please select at least one exit value.")
            
        # 1. Pre-calculate all exit scenarios
        results_list = []
        for val in exit_values:
            scenario = ExitScenario(exit_value=val, exit_date=exit_date)
            res = waterfall_model.calculate_exit_waterfall(scenario)
            results_list.append((scenario, res))
            
        # 2. Render Exit Summary Table
        if results_list:
            st.markdown("### 📊 Outcomes Summary")
            
            summary_rows = []
            
            # Use the securities from the first result as our row index
            if results_list:
                securities = list(results_list[0][1].payouts.keys())
                for sec_name in securities:
                    row = {"Security / Round": sec_name}
                    for sc, res in results_list:
                        col_name = f"${sc.exit_value:,.2f}M Exit"
                        payout = res.payouts.get(sec_name, 0.0)
                        moic = res.moic.get(sec_name, 0.0)
                        row[col_name] = f"${payout:,.2f}M ({moic:,.2f}x)"
                    summary_rows.append(row)
                    
            summary_df = pd.DataFrame(summary_rows)
            
            # Render as raw HTML to ensure easy dragging/copy-pasting into Excel
            html_table = summary_df.to_html(index=False, border=0)
            
            # Mild CSS to make the raw HTML look native to Streamlit's aesthetic
            st.markdown(
                f"""<style>
.waterfall-summary-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: "Source Sans Pro", sans-serif;
    margin-bottom: 2rem;
}}
.waterfall-summary-table th, .waterfall-summary-table td {{
    padding: 10px;
    text-align: right;
    border-bottom: 1px solid #e6e6e6;
    color: inherit;
}}
.waterfall-summary-table th {{
    background-color: transparent;
    font-weight: 600;
    border-bottom: 2px solid #e6e6e6;
}}
.waterfall-summary-table th:first-child, .waterfall-summary-table td:first-child {{
    text-align: left;
    font-weight: 600;
}}
</style>
<div style="overflow-x: auto;">
    {html_table.replace('class="dataframe"', 'class="waterfall-summary-table"')}
</div>""", 
                unsafe_allow_html=True
            )
            
            # 3. Render Detailed Breakdowns
            st.divider()
            st.markdown("### 🔎 Detailed Waterfall Breakdowns")
            
        for scenario, result in results_list:
            with st.expander(f"Exit: \\${scenario.exit_value:,.2f}M on {scenario.exit_date}", expanded=False):
                st.write("#### Payouts")
                
                # Format dataframe with commas for Currency and MOIC
                payout_data = []
                for sec_name, amount in result.payouts.items():
                    payout_data.append({
                        "Security": sec_name,
                        "Payout ($M)": amount,
                        "MOIC (x)": result.moic.get(sec_name, 0.0)
                    })
                    
                df_payouts = pd.DataFrame(payout_data)
                
                st.dataframe(
                    df_payouts.style.format({
                        "Payout ($M)": "${:,.2f}M",
                        "MOIC (x)": "{:,.2f}x"
                    }), 
                    width="stretch"
                )
                
                st.write("#### Explanations / Troubleshooting")
                # Group explanations by their tag (e.g. `[Series A]`)
                grouped_explanations = {}
                for exp in result.explanations:
                    tag, msg = "Summary", exp
                    if exp.startswith("[") and "]" in exp:
                        tag, msg = exp[1:].split("]", 1)
                        msg = msg.strip()
                    grouped_explanations.setdefault(tag, []).append(msg)
                    
                # Display grouped explanations
                for tag, msgs in grouped_explanations.items():
                    with st.expander(f"📌 {tag}", expanded=(tag == "Summary")):
                        for m in msgs:
                            escaped_msg = m.replace('$', r'\$')
                            st.write(f"- {escaped_msg}")
