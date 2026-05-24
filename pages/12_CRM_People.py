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
                
        st.divider()
        st.subheader("🏢 Organizations & Roles")
        if person.roles:
            for role in person.roles:
                org = role.organization
                if org:
                    with st.expander(f"🏢 {org.name} — {role.title or 'No Title'}"):
                        st.write(f"**Title:** {role.title or 'No Title'}")
                        
                        years = ""
                        if role.start_date:
                            start_str = role.start_date.strftime("%Y")
                            end_str = role.end_date.strftime("%Y") if role.end_date else "Present"
                            years = f"{start_str} - {end_str}"
                        elif role.end_date:
                            years = f"Until {role.end_date.strftime('%Y')}"
                            
                        st.write(f"**Years:** {years}")
                        st.write(f"**Seniority:** {role.seniority_level or 'N/A'}")
                        st.write(f"**Role Type:** {role.role_type or 'N/A'}")
                        st.write(f"**Current:** {role.is_current}")
                        st.write(f"**Source:** {role.source or 'N/A'}")
        else:
            st.info("No organizations linked to this person.")

        # Linked Source Documents
        st.divider()
        st.subheader("📄 Linked Source Documents")
        
        doc_ids_subquery = db.query(SourceDocument.id).join(DocumentText).join(ExtractionJob).join(ExtractedEntity).join(EntityMatch).filter(
            EntityMatch.canonical_entity_type == "Person",
            EntityMatch.canonical_entity_id == str(person.id)
        ).subquery()
        
        docs = db.query(SourceDocument).filter(SourceDocument.id.in_(doc_ids_subquery)).all()
        
        if docs:
            from market_comps.config import get_supabase_url
            import zoneinfo
            eastern = zoneinfo.ZoneInfo("America/New_York")
            
            for doc in docs:
                signed_url = get_supabase_url(doc.file_path) if doc.file_path else ""
                if signed_url:
                    url_display = f"{doc.source_url} [(View)]({signed_url})"
                else:
                    url_display = f"[{doc.source_url}]({doc.source_url})" if str(doc.source_url).startswith("http") else doc.source_url
                
                tz_time = doc.created_at.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(eastern).strftime('%Y-%m-%d %I:%M %p ET') if doc.created_at else "Unknown Time"
                st.markdown(f"- **{doc.document_type}**: {url_display} (Processed: {tz_time})")
        else:
            st.info("No documents linked to this person.")

        # Audit Trail
        st.divider()
        st.subheader("📜 Audit Trail")
        filters = [
            (CanonicalMutation.canonical_entity_type == "PERSON") & (CanonicalMutation.canonical_entity_id == str(person.id)),
            (CanonicalMutation.canonical_entity_type == "PERSON_EMAIL") & (CanonicalMutation.canonical_entity_id == str(person.id))
        ]
        role_ids = [str(role.id) for role in person.roles] if person.roles else []
        if role_ids:
            filters.append((CanonicalMutation.canonical_entity_type == "PERSON_ROLE") & (CanonicalMutation.canonical_entity_id.in_(role_ids)))
            
        audit = db.query(CanonicalMutation).filter(
            or_(*filters)
        ).order_by(CanonicalMutation.created_at.desc()).limit(20).all()
        
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
