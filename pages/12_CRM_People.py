import streamlit as st
import pandas as pd
import uuid
from sqlalchemy.orm import joinedload
from sqlalchemy import or_
from market_comps.db.session import get_db
from market_comps.db.models import (
    Person, PersonEmail, PersonOrganizationRole, CanonicalMutation,
    EntityMatch, ExtractedEntity, ExtractionJob, DocumentText, SourceDocument
)

st.set_page_config(page_title="People Directory", page_icon="👤", layout="wide")
st.title("👤 People Directory")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Helper to render person details below the table
def display_person_details(person_id):
    # Parse string uuid
    u_id = uuid.UUID(person_id) if isinstance(person_id, str) else person_id

    person = db.query(Person).options(
        joinedload(Person.emails),
        joinedload(Person.roles).joinedload(PersonOrganizationRole.organization)
    ).filter(Person.id == u_id).first()

    if not person:
        st.error(f"Person with ID {person_id} not found.")
        return

    with st.container(border=True):
        st.subheader(f"👤 {person.full_name or person.first_name + ' ' + person.last_name}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Basic Information")
            st.write(f"**Name:** {person.full_name or 'N/A'}")
            st.write(f"**LinkedIn:** {person.linkedin_url or 'N/A'}")
            st.write(f"**Twitter/X:** {person.twitter_url or 'N/A'}")
            st.write(f"**Location:** {person.city or 'N/A'}, {person.state or 'N/A'}, {person.country or 'N/A'}")
            if person.bio:
                st.info(person.bio)
                
            if person.emails:
                st.divider()
                st.markdown("#### Email Addresses")
                for e in person.emails:
                    primary_badge = "🌟 [Primary]" if e.is_primary else ""
                    st.write(f"- 📨 **{e.email}** ({e.email_type or 'Work'}) {primary_badge}")
                    
        with col2:
            st.markdown("#### Roles & Organizations")
            if person.roles:
                for role in person.roles:
                    org = role.organization
                    if org:
                        with st.expander(f"🏢 {org.name} — {role.title or 'No Title'}"):
                            st.write(f"**Title:** {role.title or 'No Title'}")
                            st.write(f"**Seniority:** {role.seniority_level or 'N/A'}")
                            st.write(f"**Role Type:** {role.role_type or 'N/A'}")
                            st.write(f"**Current:** {role.is_current}")
                            st.write(f"**Source:** {role.source or 'N/A'}")
            else:
                st.info("No organizations linked to this person.")

        # Linked Source Documents
        st.divider()
        st.subheader("📄 Linked Source Documents")
        
        docs = db.query(SourceDocument).join(DocumentText).join(ExtractionJob).join(ExtractedEntity).join(EntityMatch).filter(
            EntityMatch.canonical_entity_type == "Person",
            EntityMatch.canonical_entity_id == str(person.id)
        ).distinct().all()
        
        if docs:
            for doc in docs:
                st.write(f"- **{doc.document_type}**: {doc.source_url} (Processed: {doc.created_at.strftime('%Y-%m-%d')})")
        else:
            st.info("No documents linked to this person.")

        # Audit Trail
        st.divider()
        st.subheader("📜 Mutation History")
        audit = db.query(CanonicalMutation).filter_by(
            canonical_entity_type="PERSON", canonical_entity_id=str(person.id)
        ).order_by(CanonicalMutation.created_at.desc()).limit(10).all()
        
        if audit:
            audit_data = []
            for a in audit:
                audit_data.append({
                    "Date": a.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Action": a.mutation_type,
                    "Field": a.field_name or "",
                    "Old Value": a.old_value or "",
                    "New Value": a.new_value or "",
                    "Source": a.source or ""
                })
            st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)
        else:
            st.info("No mutation entries found.")

# Fetch and query data
search_query = st.text_input("Search People...", placeholder="Search by name...")

q = db.query(Person)
if search_query:
    search_filter = f"%{search_query}%"
    q = q.filter(
        or_(
            Person.full_name.ilike(search_filter),
            Person.first_name.ilike(search_filter),
            Person.last_name.ilike(search_filter)
        )
    )

people = q.order_by(Person.created_at.desc()).limit(200).all()

# Prepare Dataframe
data = []
for p in people:
    data.append({
        "ID": str(p.id),
        "Name": p.full_name or f"{p.first_name} {p.last_name}",
        "LinkedIn": p.linkedin_url,
        "City": p.city,
        "State": p.state,
        "Created": p.created_at.strftime("%Y-%m-%d") if p.created_at else ""
    })

df = pd.DataFrame(data)

if df.empty:
    st.info("No people records found.")
else:
    st.write("👆 *Select a person row below to inspect full details.*")
    
    event = st.dataframe(
        df,
        key="grid_people",
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        use_container_width=True,
        column_config={
            "ID": None, # Hide ID
            "LinkedIn": st.column_config.LinkColumn("LinkedIn"),
            "Created": st.column_config.DateColumn("Created")
        }
    )
    
    selection = event.get("selection", {})
    rows = selection.get("rows", [])
    
    if rows:
        st.divider()
        selected_row_idx = rows[0]
        selected_person_id = df.iloc[selected_row_idx]["ID"]
        display_person_details(selected_person_id)
