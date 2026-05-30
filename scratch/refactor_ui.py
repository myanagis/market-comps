with open("pages/12_CRM_Investment_Firms.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the start of the rendering logic
start_marker = "# Fetch and query data\n"
parts = content.split(start_marker)

if len(parts) == 2:
    header = parts[0]
    firms_logic = parts[1]
    
    # Indent the firms logic
    indented_firms = "\n".join("    " + line if line else "" for line in firms_logic.split("\n"))
    
    new_funds_logic = """
with tab_funds:
    st.subheader("💰 All Fund Profiles")
    
    fund_q = db.query(FundProfile).join(Organization).order_by(FundProfile.created_at.desc())
    funds = fund_q.all()
    
    if not funds:
        st.info("No funds found in the CRM.")
    else:
        fund_data = []
        for f in funds:
            fund_data.append({
                "Fund ID": f.id,
                "Investment Firm": f.parent_organization.name if f.parent_organization else "Unknown",
                "Fund Name": f.fund_name,
                "Fund Type": f.fund_type or f.investment_fund_type or "",
                "Vintage": str(f.vintage_year) if f.vintage_year else "",
                "Raised": f.fund_size_raised or "",
                "Target": f.fund_size_target or "",
                "Status": f.status or "",
            })
            
        import pandas as pd
        fund_df = pd.DataFrame(fund_data)
        st.dataframe(fund_df, use_container_width=True, hide_index=True)
"""
    
    final_content = header + "with tab_firms:\n" + "    search_query = st.text_input(\"Search Investors...\", placeholder=\"Search by name, domain, or website...\")\n" + indented_firms[4+len("search_query = st.text_input(\"Search Investors...\", placeholder=\"Search by name, domain, or website...\")"):] + "\n" + new_funds_logic
    
    # Actually wait, let's just do a simple replacement
    new_firms_code = "with tab_firms:\n" + indented_firms
    
    with open("pages/12_CRM_Investment_Firms.py", "w", encoding="utf-8") as f:
        f.write(header + new_firms_code + new_funds_logic)
    print("Successfully refactored 12_CRM_Investment_Firms.py")
else:
    print("Could not find start marker.")
