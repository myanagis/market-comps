import streamlit as st
import pandas as pd
from market_comps.waterfall.models import CapTable, SecurityClass, ExitScenario, SecurityType
from market_comps.waterfall.chat_agent import WaterfallChatAgent
from market_comps.waterfall.calculator import WaterfallCalculator

st.set_page_config(page_title="Waterfall Calculator", page_icon="💧", layout="wide")

st.title("💧 Cap Table & Waterfall Calculator")
st.markdown("""
Build your company's cap table dynamically using the chat interface below. 
You can add rounds (like "*$2M SAFE at a $10M cap*") and define exit scenarios (like "*$100M exit in 2026*").
""")

# Initialize Session State
if "cap_table" not in st.session_state:
    st.session_state.cap_table = CapTable()
if "exit_scenarios" not in st.session_state:
    st.session_state.exit_scenarios = []  # List of ExitScenario
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome! Tell me about your first funding round, or how many common shares you're starting with."}
    ]

# ── TOP: Cap Table Display ───────────────────────────────────────────────────
st.header("📊 Current Cap Table")

ct: CapTable = st.session_state.cap_table

if not ct.securities:
    st.info("Your cap table is currently empty. Use the chat below to add your first securities or founders' shares.")
else:
    # Convert CapTable to a dataframe for display
    data = []
    for s in sorted(ct.securities, key=lambda x: x.seniority):
        data.append({
            "Series": s.series_name,
            "Type": s.security_type.value.title(),
            "Close Date": s.close_date or "-",
            "Seniority": s.seniority,
            "Total $": f"${s.total_investment_usd:,.2f}" if s.total_investment_usd else "-",
            "Shares": f"{s.total_shares:,}" if s.total_shares else "-",
            "Price/Share": f"${s.issue_price:,.4f}" if s.issue_price else "-",
            "Liq Pref": f"{s.liquidation_preference_multiple}x",
            "Participating?": "Yes" if s.is_participating else "No",
            "Discount": f"{s.discount_rate * 100}%" if s.discount_rate else "-",
            "Val Cap": f"${s.valuation_cap:,.0f}" if s.valuation_cap else "-",
        })
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── MIDDLE: Exit Scenarios ───────────────────────────────────────────────────
st.header("💸 Exit Scenarios")
if not st.session_state.exit_scenarios:
    st.caption("No exit scenarios defined yet. Try asking: *'What happens in a $50M exit?'*")
else:
    for i, exit_scen in enumerate(st.session_state.exit_scenarios):
        with st.expander(f"Exit: ${exit_scen.exit_value_usd:,.0f}" + (f" on {exit_scen.exit_date}" if exit_scen.exit_date else ""), expanded=True):
            
            calc_result = WaterfallCalculator.calculate_exit_waterfall(ct, exit_scen)
            
            payout_data = []
            for sec_id, amount in calc_result.payouts.items():
                sec = next((s for s in ct.securities if s.id == sec_id), None)
                if sec:
                    payout_data.append({
                        "Series": sec.series_name,
                        "Payout": f"${amount:,.2f}"
                    })
            if payout_data:
                st.table(pd.DataFrame(payout_data))
            else:
                st.write("No configured payouts yet pending mathematical logic implementation.")

# ── BOTTOM: Chat Interface ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("💬 Chat to Edit Cap Table")

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("E.g., 'Add a $5M Series A at $20M pre-money' or 'What if we exit for $100M?'"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            agent = WaterfallChatAgent(model="openai/gpt-4o")
            try:
                action_data, reply_msg = agent.process_message(prompt, ct)
                
                # Apply the action
                action = action_data.get("action")
                
                if action == "add_security" and "security" in action_data:
                    sec_data = action_data["security"]
                    new_sec = SecurityClass(**sec_data)
                    ct.add_security(new_sec)
                    reply_msg += f"\n\n*(Added {new_sec.series_name} to Cap Table)*"
                    
                elif action == "edit_security" and "security" in action_data:
                    # Remove old and add new
                    sec_data = action_data["security"]
                    series_name = sec_data.get("series_name")
                    if series_name:
                        ct.remove_security(series_name)
                        ct.add_security(SecurityClass(**sec_data))
                        reply_msg += f"\n\n*(Updated {series_name})*"
                        
                elif action == "remove_security" and "series_to_remove" in action_data:
                    series_name = action_data["series_to_remove"]
                    success = ct.remove_security(series_name)
                    if success:
                        reply_msg += f"\n\n*(Removed {series_name})*"
                    else:
                        reply_msg += f"\n\n*(Could not find series '{series_name}' to remove)*"
                        
                elif action == "update_exit" and "exit_scenario" in action_data:
                    exit_data = action_data["exit_scenario"]
                    new_exit = ExitScenario(**exit_data)
                    st.session_state.exit_scenarios.append(new_exit)
                    reply_msg += f"\n\n*(Added Exit Scenario: ${new_exit.exit_value_usd:,.0f})*"
                    
                # Store updated state just in case
                st.session_state.cap_table = ct
                st.markdown(reply_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": reply_msg})
                
                # Force UI refresh to show new table state
                st.rerun()
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
