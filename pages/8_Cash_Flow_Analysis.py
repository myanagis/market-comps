import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
import pyxirr

st.title("Cash Flow Analysis 💸")

# Configuration Constants
SHEET_GROSS_CF = "LP Gross Cash Flows"
GROSS_CF__FUND = "Fund"
GROSS_CF__COMPANY = "Company"
GROSS_CF__DATE = "Date"
GROSS_CF__AMOUNT = "Gross Cash Flow"

SHEET_UV = "Unrealized Value"
UV__FUND = "Fund"
UV__COMPANY = "Company"
UV__DATE = "Date"
UV__AMOUNT = "Gross Cash Flow"

SHEET_UNLEV = "Unlevered Cash Flow"
UNLEV__FUND = "Fund"
UNLEV__DATE = "Date"
UNLEV__AMOUNT = "LP Net Cash Flow"

@st.cache_data(ttl=3600*24)
def get_benchmark_data(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        b_name = info.get('shortName', info.get('longName', ticker))
        
        hist = t.history(period="max")
        if not hist.empty and 'Close' in hist.columns:
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            return hist['Close'], b_name
    except Exception:
        pass
    return pd.Series(), ticker

def xirr(cashflows, dates):
    try:
        res = pyxirr.xirr(dates, cashflows)
        return float(res) if res is not None else np.nan
    except Exception:
        return np.nan

def format_currency(val):
    if pd.isna(val) or val is None:
        return "$0"
    abs_val = abs(val)
    if abs_val >= 1_000_000_000:
        return f"${val/1_000_000_000:.1f}B"
    elif abs_val >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    elif abs_val >= 1_000:
        return f"${val/1_000:.1f}K"
    else:
        return f"${val:,.0f}"

def calculate_pme(dates, cashflows, benchmark_data, calc_end_date, benchmark_ticker):
    ks_pme = np.nan
    direct_alpha = np.nan
    pme_cfs = None
    
    if benchmark_data is None or benchmark_data.empty or calc_end_date is None:
        return ks_pme, direct_alpha, pme_cfs
        
    try:
        pme_cfs = pd.DataFrame({'Date': dates, 'Cash Flow': cashflows})
        pme_cfs['Date'] = pd.to_datetime(pme_cfs['Date'])
        calc_end_ts = pd.to_datetime(calc_end_date)
        
        def get_index_val(dt):
            if dt in benchmark_data.index:
                return benchmark_data.loc[dt]
            prior_dates = benchmark_data.index[benchmark_data.index <= dt]
            if len(prior_dates) > 0:
                return benchmark_data.loc[prior_dates[-1]]
            if len(benchmark_data) > 0:
                return benchmark_data.iloc[0]
            return np.nan
            
        index_end_val = get_index_val(calc_end_ts)
        
        if not pd.isna(index_end_val):
            pme_cfs['Benchmark Value'] = pme_cfs['Date'].apply(get_index_val)
            pme_cfs['Growth Factor'] = index_end_val / pme_cfs['Benchmark Value']
            pme_cfs[f'{benchmark_ticker} CF'] = pme_cfs['Cash Flow'] * pme_cfs['Growth Factor']
            
            sum_fv_inv = abs(pme_cfs[pme_cfs['Cash Flow'] < 0][f'{benchmark_ticker} CF'].sum())
            sum_fv_dist = pme_cfs[pme_cfs['Cash Flow'] > 0][f'{benchmark_ticker} CF'].sum()
            
            if sum_fv_inv > 0:
                ks_pme = sum_fv_dist / sum_fv_inv
                
            direct_alpha = xirr(pme_cfs[f'{benchmark_ticker} CF'].tolist(), pme_cfs['Date'].tolist())
    except Exception as e:
        st.warning(f"Error computing PME: {e}")
        
    return ks_pme, direct_alpha, pme_cfs

def render_performance_metrics(fund, group, df_uv, date_col, amount_col, title_prefix="Performance Metrics", benchmark_ticker=None, benchmark_name=None, calc_end_date=None, benchmark_data=None):
    if group.empty:
        st.markdown(f"#### 📊 {title_prefix}: **{fund}**")
        st.info("No data available for this fund.")
        return
        
    inv_group = group[group[amount_col] < 0]
    dist_group = group[group[amount_col] > 0]
    
    total_inv = abs(inv_group[amount_col].sum())
    total_dist = dist_group[amount_col].sum()
    earliest_inv = inv_group[date_col].min()
    vintage = earliest_inv.year if not pd.isna(earliest_inv) else "N/A"
    latest_inv = inv_group[date_col].max()
    
    st.markdown(f"#### 📊 {title_prefix}: **{fund} ({vintage})**")
    
    cashflows = group[amount_col].tolist()
    dates = group[date_col].tolist()
    
    uv_fund = df_uv[df_uv[UV__FUND] == fund] if UV__FUND in df_uv.columns else pd.DataFrame()
    val_col = UV__AMOUNT if UV__AMOUNT in df_uv.columns else 'Unrealized Value'
    
    total_uv = 0.0
    if len(uv_fund) > 0 and val_col in uv_fund.columns:
        latest_uvs = uv_fund.sort_values(UV__DATE).groupby(UV__COMPANY).tail(1)
        for _, row in latest_uvs.iterrows():
            uv = row[val_col]
            if not pd.isna(uv) and uv > 0:
                total_uv += uv
                cashflows.append(uv)
                dates.append(row[UV__DATE] if not pd.isna(row[UV__DATE]) else pd.Timestamp.today())
                
    irr = xirr(cashflows, dates)
    dpi = (total_dist / total_inv) if total_inv > 0 else 0
    moic = ((total_dist + total_uv) / total_inv) if total_inv > 0 else 0
    
    # Calculate KS-PME and Direct Alpha Native
    b_label = benchmark_name if benchmark_name else benchmark_ticker
    ks_pme, direct_alpha, pme_cfs = calculate_pme(dates, cashflows, benchmark_data, calc_end_date, b_label)

    m_cols1 = st.columns(4)
    with m_cols1[0].container(border=True):
        st.metric("Vintage", str(vintage))
    with m_cols1[1].container(border=True):
        st.metric("Total Invested", format_currency(total_inv))
    with m_cols1[2].container(border=True):
        st.metric("Total Distrib.", format_currency(total_dist))
    with m_cols1[3].container(border=True):
        st.metric("Unrealized Value", format_currency(total_uv))

    m_cols2 = st.columns(4)
    with m_cols2[0].container(border=True):
        st.metric("DPI", f"{dpi:.2f}x")
    with m_cols2[1].container(border=True):
        st.metric("MOIC", f"{moic:.2f}x")
    with m_cols2[2].container(border=True):
        st.metric("IRR", f"{irr:.2%}" if not pd.isna(irr) else "N/A")
    with m_cols2[3].container(border=True):
        st.metric("Latest Invesmt.", latest_inv.strftime('%m/%Y') if not pd.isna(latest_inv) else "N/A")
        
    if benchmark_data is not None and not benchmark_data.empty and calc_end_date is not None:
        m_cols3 = st.columns(2)
        with m_cols3[0].container(border=True):
            st.metric(f"KS-PME ({b_label})", f"{ks_pme:.2f}x" if not pd.isna(ks_pme) else "N/A")
        with m_cols3[1].container(border=True):
            st.metric(f"Direct Alpha ({b_label})", f"{direct_alpha:.2%}" if not pd.isna(direct_alpha) else "N/A")
        
    with st.expander(f"View Raw IRR Calculation Inputs for {fund}"):
        if pme_cfs is not None and not pme_cfs.empty and 'Growth Factor' in pme_cfs.columns:
            irr_df = pme_cfs.sort_values('Date').copy()
            irr_df['Date'] = irr_df['Date'].dt.strftime('%m/%d/%Y').fillna('-')
            st.dataframe(irr_df.style.format({
                'Cash Flow': '${:,.2f}', 
                'Benchmark Value': '{:,.2f}', 
                'Growth Factor': '{:.4f}x', 
                f'{b_label} CF': '${:,.2f}'
            }))
        else:
            irr_df = pd.DataFrame({'Date': dates, 'Cash Flow': cashflows}).sort_values('Date')
            irr_df['Date'] = irr_df['Date'].dt.strftime('%m/%d/%Y').fillna('-')
            st.dataframe(irr_df.style.format({'Cash Flow': '${:,.2f}'}))

def render_cashflow_chart(df_flows, date_col, amount_col, freq='Y'):
    if df_flows.empty:
        return
    
    df_flows['DatePeriod'] = df_flows[date_col].dt.to_period(freq).dt.start_time
    df_flows['Color'] = np.where(df_flows[amount_col] >= 0, 'Distribution', 'Investment')
    
    chart_df = df_flows.groupby(['DatePeriod', 'Color'])[amount_col].sum().reset_index()
    net_df = df_flows.groupby('DatePeriod')[amount_col].sum().reset_index()
    net_df['Color'] = 'Net Cash Flow'
    
    color_scale = alt.Scale(
        domain=['Distribution', 'Investment', 'Net Cash Flow'], 
        range=['#1f77b4', '#d62728', 'green']
    )
    
    bars = alt.Chart(chart_df).mark_bar(size=18).encode(
        x=alt.X('year(DatePeriod):O', title='Year', axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f'{amount_col}:Q', title='$ Amount'),
        color=alt.Color('Color:N', scale=color_scale),
        tooltip=[alt.Tooltip('year(DatePeriod):O', title='Year'), 'Color:N', f'{amount_col}:Q']
    )
    
    line = alt.Chart(net_df).mark_line(color='green', strokeWidth=3).encode(
        x=alt.X('year(DatePeriod):O', title='Year', axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f'{amount_col}:Q'),
        color=alt.Color('Color:N', scale=color_scale),
        tooltip=[alt.Tooltip('year(DatePeriod):O', title='Year'), 'Color:N', alt.Tooltip(f'{amount_col}:Q', title='Net Cash Flow Amount')]
    )
    
    st.altair_chart(alt.layer(bars, line).properties(height=500, width=750))


uploaded_file = st.file_uploader("Upload Cash Flows Excel File (.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    try:
        xl = pd.ExcelFile(uploaded_file)
        
        df_gross = xl.parse(SHEET_GROSS_CF)
        df_uv = xl.parse(SHEET_UV)
        df_unlev = xl.parse(SHEET_UNLEV)
        
        for df in [df_gross, df_uv, df_unlev]:
            df.columns = df.columns.str.strip()
            
        if GROSS_CF__DATE in df_gross.columns:
            df_gross[GROSS_CF__DATE] = pd.to_datetime(df_gross[GROSS_CF__DATE], errors='coerce')
        if UV__DATE in df_uv.columns:
            df_uv[UV__DATE] = pd.to_datetime(df_uv[UV__DATE], errors='coerce')
        if UNLEV__DATE in df_unlev.columns:
            df_unlev[UNLEV__DATE] = pd.to_datetime(df_unlev[UNLEV__DATE], errors='coerce')
                
        st.success("Successfully loaded Excel file!")
        
        tab1, tab2 = st.tabs(["Gross Cash Flows", "Unlevered Cash Flows"])
        
        with tab1:
            st.header("LP Gross Cash Flows")
            
            options_fund = df_gross[GROSS_CF__FUND].dropna().unique() if GROSS_CF__FUND in df_gross.columns else []
            
            fund_sel = st.selectbox('Fund Filter', options=options_fund) if len(options_fund) > 0 else None
            fund_filter = [fund_sel] if fund_sel else []
            
            st.divider()
            if len(fund_filter) > 0 and GROSS_CF__FUND in df_gross.columns:
                for fund in fund_filter:
                    group = df_gross[df_gross[GROSS_CF__FUND] == fund]
                    render_performance_metrics(fund, group, df_uv, GROSS_CF__DATE, GROSS_CF__AMOUNT, "Gross Metrics")
            
            mask = df_gross[GROSS_CF__FUND].isin(fund_filter) if GROSS_CF__FUND in df_gross.columns else pd.Series([False]*len(df_gross))
            filtered_gross = df_gross[mask].copy()
            
            if not filtered_gross.empty:
                render_cashflow_chart(filtered_gross, GROSS_CF__DATE, GROSS_CF__AMOUNT)
                
            st.divider()
            st.subheader("Company Summary Table")
            summary_res = []
            
            if GROSS_CF__FUND in df_gross.columns and GROSS_CF__COMPANY in df_gross.columns:
                for (fund, comp), group in df_gross.groupby([GROSS_CF__FUND, GROSS_CF__COMPANY]):
                    if fund not in fund_filter:
                        continue
                        
                    inv_group = group[group[GROSS_CF__AMOUNT] < 0]
                    dist_group = group[group[GROSS_CF__AMOUNT] > 0]
                    
                    inv = abs(inv_group[GROSS_CF__AMOUNT].sum())
                    dist = dist_group[GROSS_CF__AMOUNT].sum()
                    
                    earliest_inv = inv_group[GROSS_CF__DATE].min() if not inv_group.empty else pd.NaT
                    last_dist = dist_group[GROSS_CF__DATE].max() if not dist_group.empty else pd.NaT
                    
                    uv_rows = df_uv[(df_uv[UV__FUND] == fund) & (df_uv[UV__COMPANY] == comp)]
                    latest_uv = 0.0
                    max_uv_date = pd.Timestamp.today()
                    
                    val_col = UV__AMOUNT if UV__AMOUNT in df_uv.columns else 'Unrealized Value'
                    
                    if len(uv_rows) > 0 and val_col in uv_rows.columns:
                        latest_uv_row = uv_rows.sort_values(UV__DATE).iloc[-1]
                        latest_uv = latest_uv_row[val_col]
                        max_uv_date = latest_uv_row[UV__DATE]
                    
                    exited = "Yes" if (latest_uv == 0.0 or pd.isna(latest_uv)) and dist > 0 else "No"
                    
                    cashflows = group[GROSS_CF__AMOUNT].tolist()
                    dates = group[GROSS_CF__DATE].tolist()
                    if latest_uv > 0:
                        cashflows.append(latest_uv)
                        dates.append(max_uv_date)
                        
                    irr = xirr(cashflows, dates)
                    
                    summary_res.append({
                        'Fund': fund, 
                        'Company': comp, 
                        'Earliest Investment': earliest_inv,
                        'Last Distribution': last_dist,
                        'Total Investment': inv, 
                        'Total Distribution': dist, 
                        'Unrealized Value': latest_uv,
                        'Exited': exited, 
                        'Gross IRR': irr
                    })
                
            if summary_res:
                summary_df = pd.DataFrame(summary_res)
                summary_df['Earliest Investment'] = summary_df['Earliest Investment'].dt.strftime('%m/%d/%Y').fillna('-')
                summary_df['Last Distribution'] = summary_df['Last Distribution'].dt.strftime('%m/%d/%Y').fillna('-')
                
                st.dataframe(summary_df.style.format({
                    'Total Investment': '${:,.2f}', 
                    'Total Distribution': '${:,.2f}', 
                    'Unrealized Value': '${:,.2f}',
                    'Gross IRR': '{:.2%}'
                }))
            else:
                st.info("No data for current filters.")
                
            st.divider()
            st.subheader("Detailed Company Data")
            options_detailed = df_gross[GROSS_CF__COMPANY].dropna().unique() if GROSS_CF__COMPANY in df_gross.columns else []
            
            if len(options_detailed) > 0:
                options_detailed = sorted(options_detailed, key=lambda x: str(x).lower())
                det_comp = st.selectbox("Select Company to View Detailed Logs", options=options_detailed)
                
                comp_funds = df_gross[df_gross[GROSS_CF__COMPANY] == det_comp][GROSS_CF__FUND].dropna().unique() if GROSS_CF__FUND in df_gross.columns else []
                
                if len(comp_funds) > 0:
                    for cf in comp_funds:
                        c_group = df_gross[(df_gross[GROSS_CF__COMPANY] == det_comp) & (df_gross[GROSS_CF__FUND] == cf)]
                        
                        c_inv_group = c_group[c_group[GROSS_CF__AMOUNT] < 0]
                        c_inv = abs(c_inv_group[GROSS_CF__AMOUNT].sum())
                        c_dist = c_group[c_group[GROSS_CF__AMOUNT] > 0][GROSS_CF__AMOUNT].sum()
                        c_earliest_inv = c_inv_group[GROSS_CF__DATE].min()
                        c_vintage = c_earliest_inv.year if not pd.isna(c_earliest_inv) else "N/A"
                        
                        c_uv_rows = df_uv[(df_uv[UV__FUND] == cf) & (df_uv[UV__COMPANY] == det_comp)]
                        val_col = UV__AMOUNT if UV__AMOUNT in df_uv.columns else 'Unrealized Value'
                        c_uv = 0.0
                        c_max_date = pd.Timestamp.today()
                        if len(c_uv_rows) > 0 and val_col in c_uv_rows.columns:
                            c_uv_row = c_uv_rows.sort_values(UV__DATE).iloc[-1]
                            c_uv = c_uv_row[val_col]
                            c_max_date = c_uv_row[UV__DATE]
                            
                        c_cashflows = c_group[GROSS_CF__AMOUNT].tolist()
                        c_dates = c_group[GROSS_CF__DATE].tolist()
                        if c_uv > 0:
                            c_cashflows.append(c_uv)
                            c_dates.append(c_max_date if not pd.isna(c_max_date) else pd.Timestamp.today())
                            
                        c_irr = xirr(c_cashflows, c_dates)
                        
                        c_cols = st.columns(5)
                        with c_cols[0].container(border=True):
                            st.metric("Vintage", str(c_vintage))
                        with c_cols[1].container(border=True):
                            st.metric("Total Invested", format_currency(c_inv))
                        with c_cols[2].container(border=True):
                            st.metric("Total Distributed", format_currency(c_dist))
                        with c_cols[3].container(border=True):
                            st.metric("Unrealized Value", format_currency(c_uv))
                        with c_cols[4].container(border=True):
                            st.metric("Gross IRR", f"{c_irr:.2%}" if not pd.isna(c_irr) else "N/A")
                
                st.markdown(f"**Gross Cash Flows log for _{det_comp}_**")
                comp_gross = df_gross[df_gross[GROSS_CF__COMPANY] == det_comp]
                if GROSS_CF__DATE in comp_gross.columns:
                    comp_gross = comp_gross.sort_values(GROSS_CF__DATE)
                if GROSS_CF__AMOUNT in comp_gross.columns:
                    st.dataframe(comp_gross.style.format({GROSS_CF__AMOUNT: '${:,.2f}'}) if not comp_gross.empty else comp_gross)
                else:
                    st.dataframe(comp_gross)
                
                st.markdown(f"**Unrealized Value log for _{det_comp}_**")
                comp_uv = df_uv[df_uv[UV__COMPANY] == det_comp]
                if UV__DATE in comp_uv.columns:
                    comp_uv = comp_uv.sort_values(UV__DATE)
                
                val_col = UV__AMOUNT if UV__AMOUNT in comp_uv.columns else 'Unrealized Value'
                if val_col in comp_uv.columns:
                    st.dataframe(comp_uv.style.format({val_col: '${:,.2f}'}) if not comp_uv.empty else comp_uv)
                else:
                    st.dataframe(comp_uv)
            else:
                st.info("No detailed companies found to display.")
                
        with tab2:
            st.header("Unlevered Cash Flow")
            
            st.markdown("#### 📈 Benchmark KS-PME / Direct Alpha Settings")
            b_cols = st.columns(2)
            with b_cols[0]:
                benchmark_ticker = st.text_input("Benchmark Ticker (e.g., ^GSPC for S&P 500)", value='^GSPC')
            with b_cols[1]:
                calc_end_date = st.date_input("Calculation End Date", value=pd.to_datetime('today'))
                
            benchmark_data = None
            benchmark_name = benchmark_ticker
            if benchmark_ticker:
                with st.spinner("Fetching underlying Benchmark index via Yahoo Finance..."):
                    benchmark_data, benchmark_name = get_benchmark_data(benchmark_ticker)
            
            st.divider()
            
            options_unlev = df_unlev[UNLEV__FUND].dropna().unique() if UNLEV__FUND in df_unlev.columns else []
            fund_sel_unlev = st.selectbox('Fund Filter', options=options_unlev, key="unlev_fund") if len(options_unlev) > 0 else None
            fund_filter_unlev = [fund_sel_unlev] if fund_sel_unlev else []
            
            st.divider()
            if len(fund_filter_unlev) > 0:
                for fund in fund_filter_unlev:
                    group_unlev = df_unlev[df_unlev[UNLEV__FUND] == fund] if UNLEV__FUND in df_unlev.columns else pd.DataFrame()
                    render_performance_metrics(fund, group_unlev, df_uv, UNLEV__DATE, UNLEV__AMOUNT, "Unlevered Metrics", benchmark_ticker, benchmark_name, calc_end_date, benchmark_data)

            mask_unlev = df_unlev[UNLEV__FUND].isin(fund_filter_unlev) if UNLEV__FUND in df_unlev.columns else pd.Series([False]*len(df_unlev))
            filtered_unlev = df_unlev[mask_unlev].copy()
            
            if not filtered_unlev.empty:
                render_cashflow_chart(filtered_unlev, UNLEV__DATE, UNLEV__AMOUNT)

    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
