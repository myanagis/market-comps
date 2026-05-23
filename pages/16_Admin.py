import streamlit as st
import pandas as pd
from sqlalchemy import MetaData, Table, select, func, or_
from market_comps.db.session import engine

# Double check authentication
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.error("🔒 Unauthorized access. Please return to the homepage to log in.")
    st.stop()

# Page configuration
st.set_page_config(page_title="Admin DB Manager", page_icon="🔒", layout="wide")

st.title("🔒 Admin Database Manager")

# 1. Reflect Database Metadata
@st.cache_data(ttl=60)
def get_db_tables():
    """Reflect the database schema and get sorted table names."""
    try:
        metadata = MetaData()
        metadata.reflect(bind=engine)
        return sorted(list(metadata.tables.keys()))
    except Exception as e:
        st.error(f"Failed to reflect database schema: {e}")
        return []

tables = get_db_tables()

if not tables:
    st.info("No tables found in the database or connection failed.")
    st.stop()

# 2. Sidebar Table Selection & Configuration
with st.sidebar:
    st.header("Database Config")
    selected_table_name = st.selectbox("📂 Select Database Table", tables, index=0)
    
    # Query parameters
    limit = st.slider("Row Display Limit", min_value=10, max_value=5000, value=500, step=50)

# 3. Load Selected Table Schema & Stats
metadata = MetaData()
try:
    selected_table = Table(selected_table_name, metadata, autoload_with=engine)
except Exception as e:
    st.error(f"Failed to load table '{selected_table_name}': {e}")
    st.stop()

# Columns and primary key detection
columns_info = []
string_columns = []
for col in selected_table.columns:
    columns_info.append({
        "Name": col.name,
        "Type": str(col.type),
        "Primary Key": col.primary_key,
        "Nullable": col.nullable
    })
    # If the column type supports text searching
    if hasattr(col.type, "python_type") and col.type.python_type is str:
        string_columns.append(col)

# Get row count
@st.cache_data(ttl=10)
def get_row_count(table_name):
    try:
        with engine.connect() as conn:
            query = select(func.count()).select_from(Table(table_name, MetaData(), autoload_with=engine))
            result = conn.execute(query).scalar()
            return result
    except Exception:
        return 0

row_count = get_row_count(selected_table_name)

# 4. Stat Cards Row
col1, col2, col3 = st.columns(3)
col1.metric("Selected Table", selected_table_name)
col2.metric("Total Records", f"{row_count:,}")
col3.metric("Columns Count", len(columns_info))

# 5. Search & Filter Interface
st.subheader("🔍 Filters & Search")
col_search, col_sort, col_order = st.columns([2, 1, 1])

with col_search:
    search_query = st.text_input("Global Search (matches text columns)", placeholder="Type search term...")

with col_sort:
    sort_column = st.selectbox(
        "Sort By Column", 
        ["[None]"] + [col["Name"] for col in columns_info],
        index=0
    )

with col_order:
    sort_order = st.selectbox("Sort Order", ["Ascending", "Descending"], index=0)

# Build query
query = select(selected_table)

# Apply global text search filter
if search_query and string_columns:
    search_filters = [col.ilike(f"%{search_query}%") for col in string_columns]
    query = query.where(or_(*search_filters))

# Apply sorting
if sort_column != "[None]":
    sort_col_obj = getattr(selected_table.c, sort_column)
    if sort_order == "Descending":
        query = query.order_by(sort_col_obj.desc())
    else:
        query = query.order_by(sort_col_obj.asc())

# Apply limit
query = query.limit(limit)

# Load data into pandas
try:
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
except Exception as e:
    st.error(f"Error executing query: {e}")
    st.stop()

# 6. Display Table Data & Operations
tab_data, tab_schema = st.tabs(["📊 Table Data Explorer", "📋 Table Schema & Columns"])

with tab_data:
    if df.empty:
        st.info("No records matched your search/filters.")
    else:
        # Display table controls
        st.write(f"Showing top **{len(df)}** rows of **{row_count:,}** total.")
        
        # Smart link-column formatting for fields with URL values
        column_configs = {}
        for col_name in df.columns:
            # Check if column is a website or URL path
            if "url" in col_name.lower() or "website" in col_name.lower():
                column_configs[col_name] = st.column_config.LinkColumn(col_name)
        
        # Display the data frame using interactive Streamlit dataframe component
        st.dataframe(
            df,
            use_container_width=True,
            column_config=column_configs,
            hide_index=True
        )
        
        # Export options
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download current view as CSV",
            data=csv_data,
            file_name=f"{selected_table_name}_export.csv",
            mime="text/csv",
            use_container_width=False
        )

with tab_schema:
    st.write("Below are the detail definitions of all fields in this table reflected directly from the database schema.")
    df_columns = pd.DataFrame(columns_info)
    st.dataframe(df_columns, use_container_width=True, hide_index=True)
