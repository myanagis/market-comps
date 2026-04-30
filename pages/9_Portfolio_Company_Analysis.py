import streamlit as st
import pandas as pd
import altair as alt

st.title("Portfolio Company Analysis 🏢")

st.markdown("""
Upload your specific Company Data file containing tracking metrics directly mapped categorically per target Fund.

Expected fields: `Fund`, `Company`, `Entry Date`, `Exit Date`, `Fund Capital Invested`, `Realized Value`, `Unrealized Value`, `Total Value`, `Gross MOIC`, `Gross IRR`, `Industry`
""")

uploaded_file = st.file_uploader("Upload Portfolio Company Excel File (.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        # Strip all incoming column names natively
        df.columns = df.columns.str.strip()
        
        # Ensure Entry Date evaluates into mathematical DateTime arrays 
        if 'Entry Date' in df.columns:
            df['Entry Date'] = pd.to_datetime(df['Entry Date'], errors='coerce')
            
        if 'Fund' not in df.columns:
            df['Fund'] = "Unknown"
            
        st.success("Successfully loaded Portfolio Company analytics data!")
        
        st.divider()
        st.subheader("Gross MOIC by Fund & Industry")
        
        # Filter NaNs explicitly for cleanly mapping the Y-Axis specifically
        if 'Gross MOIC' in df.columns:
            chart_df = df.copy()
            # Clean numerical constraints mathematically guaranteeing quantitative charting capability 
            chart_df['Gross MOIC'] = pd.to_numeric(chart_df['Gross MOIC'].astype(str).str.replace('x', '', regex=False).str.replace(',', ''), errors='coerce')
            
            if 'Fund Capital Invested' in chart_df.columns:
                chart_df['Fund Capital Invested'] = pd.to_numeric(chart_df['Fund Capital Invested'].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
                
            if 'Entry Date' in chart_df.columns:
                chart_df['Entry Date Str'] = chart_df['Entry Date'].dt.strftime('%Y-%m-%d').astype(str).replace('nan', 'N/A')
                
            # Allow charts to safely drop NaNs selectively based strictly on their own targets natively
            moic_df = chart_df.dropna(subset=['Gross MOIC']).copy()
            
            if not moic_df.empty:
                
                ttips = [alt.Tooltip('Company:N'), alt.Tooltip('Fund:N')]
                if 'Industry' in chart_df.columns: ttips.append(alt.Tooltip('Industry:N'))
                if 'Entry Date Str' in chart_df.columns: ttips.append(alt.Tooltip('Entry Date Str:N', title='Entry Date'))
                if 'Fund Capital Invested' in chart_df.columns: ttips.append(alt.Tooltip('Fund Capital Invested:Q', format='$,.0f'))
                ttips.append(alt.Tooltip('Gross MOIC:Q', format='.2f'))
                
                c_enc = alt.Color('Industry:N', title='Industry Sector', scale=alt.Scale(scheme='set2')) if 'Industry' in chart_df.columns else alt.value('steelblue')
                s_enc = alt.Size('Fund Capital Invested:Q', title='Capital Invested', scale=alt.Scale(range=[50, 1500])) if 'Fund Capital Invested' in chart_df.columns else alt.value(200)

                # Plot each company dot mathematically evaluating `Size` locally against Invested Capital and `Color` categorically against Industry
                base_chart = alt.Chart(moic_df).mark_circle(opacity=0.6).encode(
                    x=alt.X('Fund:N', title='Target Fund', axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=0, labelExpr="split(datum.value, ' ')")),
                    y=alt.Y('Gross MOIC:Q', title='Gross MOIC', scale=alt.Scale(zero=True)),
                    size=s_enc,
                    color=c_enc,
                    tooltip=ttips
                ).properties(
                    height=550,
                    width=800
                )
                
                st.altair_chart(base_chart, use_container_width=True)
                
                st.divider()
                st.subheader("Isolated Metric Scatter Distributions")
                
                scols = st.columns(2)
                
                # Chart 1: MOIC Isolated natively (Fixed Constant Radius)
                moic_chart = alt.Chart(moic_df).mark_circle(opacity=0.7, size=150).encode(
                    x=alt.X('Fund:N', title='Target Fund', axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=0, labelExpr="split(datum.value, ' ')")),
                    y=alt.Y('Gross MOIC:Q', title='Gross MOIC', scale=alt.Scale(zero=True)),
                    color=c_enc,
                    tooltip=ttips
                ).properties(height=450)
                
                with scols[0]:
                    st.altair_chart(moic_chart, use_container_width=True)
                    
                # Chart 2: Capital Invested Isolated (Fixed Constant Radius)
                if 'Fund Capital Invested' in chart_df.columns:
                    spent_df = chart_df.dropna(subset=['Fund Capital Invested']).copy()
                    spent_chart = alt.Chart(spent_df).mark_circle(opacity=0.7, size=150).encode(
                        x=alt.X('Fund:N', title='Target Fund', axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=0, labelExpr="split(datum.value, ' ')")),
                        y=alt.Y('Fund Capital Invested:Q', title='Capital Invested', axis=alt.Axis(format='$,.0f')),
                        color=c_enc,
                        tooltip=ttips
                    ).properties(height=450)
                    with scols[1]:
                        st.altair_chart(spent_chart, use_container_width=True)
                
                # --- NEW: Fundamental & Leverage Analysis Graphs ---
                st.divider()
                st.subheader("Fundamental & Leverage Metrics")
                
                extra_cols = [
                    'EV/EBITDA (LTM)',
                    'Entry Multiples EV/EBITDA (forward)',
                    'EBITDA (LTM)',
                    'EBITDA (NTM)',
                    'LTV at Purchase (Debt/TEV)',
                    'Leverage (Debt/EBITDA)'
                ]
                
                available_extra_cols = [c for c in extra_cols if c in df.columns]
                
                if len(available_extra_cols) > 0:
                    fcols = st.columns(2)
                    
                    with fcols[0]:
                        metric_1 = st.selectbox("Select Metric 1", options=available_extra_cols, index=0)
                    with fcols[1]:
                        metric_2 = st.selectbox("Select Metric 2", options=available_extra_cols, index=min(1, len(available_extra_cols)-1))
                        
                    for col_name in [metric_1, metric_2]:
                        if col_name in chart_df.columns:
                            chart_df[col_name] = pd.to_numeric(
                                chart_df[col_name].astype(str).str.replace(r'[xX$,%]', '', regex=True), 
                                errors='coerce'
                            )
                            
                    # Render Graph 1 dynamically dropping NA rows locally
                    df_m1 = chart_df.dropna(subset=[metric_1]).copy()
                    if not df_m1.empty:
                        ttips1 = ttips.copy() + [alt.Tooltip(f'{metric_1}:Q', format=',.2f')]
                        g1_chart = alt.Chart(df_m1).mark_circle(opacity=0.7, size=150).encode(
                            x=alt.X('Fund:N', title='Target Fund', axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=0, labelExpr="split(datum.value, ' ')")),
                            y=alt.Y(f'{metric_1}:Q', title=metric_1, scale=alt.Scale(zero=False)),
                            color=c_enc,
                            tooltip=ttips1
                        ).properties(height=450)
                        
                        with fcols[0]:
                            st.altair_chart(g1_chart, use_container_width=True)
                            
                    # Render Graph 2
                    df_m2 = chart_df.dropna(subset=[metric_2]).copy()
                    if not df_m2.empty:
                        ttips2 = ttips.copy() + [alt.Tooltip(f'{metric_2}:Q', format=',.2f')]
                        g2_chart = alt.Chart(df_m2).mark_circle(opacity=0.7, size=150).encode(
                            x=alt.X('Fund:N', title='Target Fund', axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=0, labelExpr="split(datum.value, ' ')")),
                            y=alt.Y(f'{metric_2}:Q', title=metric_2, scale=alt.Scale(zero=False)),
                            color=c_enc,
                            tooltip=ttips2
                        ).properties(height=450)
                        
                        with fcols[1]:
                            st.altair_chart(g2_chart, use_container_width=True)
                
                with st.expander(f"View Active Raw Portfolio Dataset Mapping"):
                    out_df = df.copy()
                    if 'Entry Date' in out_df.columns:
                        out_df['Entry Date'] = out_df['Entry Date'].dt.strftime('%m/%d/%Y').fillna('-')
                    if 'Exit Date' in out_df.columns:
                        try:
                            out_df['Exit Date'] = pd.to_datetime(out_df['Exit Date'], errors='coerce').dt.strftime('%m/%d/%Y').fillna('-')
                        except Exception:
                            pass
                            
                    st.dataframe(out_df)
            else:
                st.warning("Insufficient or missing numerical 'Gross MOIC' coordinates detected.")
        else:
             st.error("Missing mandatory 'Gross MOIC' column in parsed dataset.")
            
    except Exception as e:
        st.error(f"Failed to internally parse target structure: {e}")
