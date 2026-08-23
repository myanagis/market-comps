import streamlit as st
from market_comps.db.models import Base

st.set_page_config(page_title="Database Schema ERD", page_icon="🗄️", layout="wide")
st.title("🗄️ Database Schema (ERD)")

st.markdown("This Entity-Relationship Diagram is generated dynamically from the underlying SQLAlchemy metadata. It automatically updates whenever new tables or relationships are added.")

dot_str = ["digraph ERD {"]
dot_str.append("  node [shape=record, fontname=Helvetica, fontsize=10];")
dot_str.append("  edge [fontname=Helvetica, fontsize=10];")
dot_str.append("  rankdir=LR;")

# Build nodes
for table_name, table in Base.metadata.tables.items():
    cols = []
    for col in table.columns:
        pk = " (PK)" if col.primary_key else ""
        fk = " (FK)" if col.foreign_keys else ""
        cols.append(f"{col.name} : {col.type}{pk}{fk}")
    
    label = "{" + f"{table_name}|" + "\\l".join(cols) + "\\l" + "}"
    dot_str.append(f'  "{table_name}" [label="{label}"];')

# Build edges
for table_name, table in Base.metadata.tables.items():
    for fk in table.foreign_keys:
        target_table = fk.column.table.name
        dot_str.append(f'  "{table_name}" -> "{target_table}" [label="{fk.parent.name} -> {fk.column.name}"];')

dot_str.append("}")

dot_code = "\n".join(dot_str)

with st.expander("View DOT Source"):
    st.code(dot_code, language="dot")

st.graphviz_chart(dot_code, use_container_width=True)
