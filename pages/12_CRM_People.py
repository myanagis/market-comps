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
from market_comps.ingestion.reconciler import log_mutation

st.set_page_config(page_title="People Directory", page_icon="👤", layout="wide")
st.title("👤 People Directory")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

@st.dialog("Edit Person")
def edit_person_dialog(person):
    with st.form("edit_person"):
        st.write(f"Edit details for **{person.full_name or person.first_name + ' ' + person.last_name}**")
        
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name", value=person.first_name or "")
            last_name = st.text_input("Last Name", value=person.last_name or "")
            linkedin = st.text_input("LinkedIn URL", value=person.linkedin_url or "")
            twitter = st.text_input("Twitter URL", value=person.twitter_url or "")
        with col2:
            city = st.text_input("City", value=person.city or "")
            state = st.text_input("State", value=person.state or "")
            country = st.text_input("Country", value=person.country or "")
            
        bio = st.text_area("Bio", value=person.bio or "")
            
        if st.form_submit_button("Save Changes"):
            user = st.session_state.get("user_email", "SYSTEM")
            
            def check_and_update(entity_type, entity_id, field_name, old_val, new_val, obj):
                if str(old_val) != str(new_val) and (old_val or new_val):
                    log_mutation(
                        db, entity_type, entity_id, "UPDATE",
                        field_name=field_name,
                        old_value=str(old_val),
                        new_value=str(new_val),
                        source="USER_EDIT",
                        created_by=user
                    )
                    setattr(obj, field_name, new_val)
                    
            check_and_update("PERSON", str(person.id), "first_name", person.first_name, first_name, person)
            check_and_update("PERSON", str(person.id), "last_name", person.last_name, last_name, person)
            full_name = f"{first_name} {last_name}".strip()
            check_and_update("PERSON", str(person.id), "full_name", person.full_name, full_name, person)
            check_and_update("PERSON", str(person.id), "linkedin_url", person.linkedin_url, linkedin, person)
            check_and_update("PERSON", str(person.id), "twitter_url", person.twitter_url, twitter, person)
            check_and_update("PERSON", str(person.id), "city", person.city, city, person)
            check_and_update("PERSON", str(person.id), "state", person.state, state, person)
            check_and_update("PERSON", str(person.id), "country", person.country, country, person)
            check_and_update("PERSON", str(person.id), "bio", person.bio, bio, person)
            
            db.commit()
            st.rerun()

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
        col_h1, col_h2 = st.columns([5, 1])
        with col_h1:
            st.subheader(f"👤 {person.full_name or person.first_name + ' ' + person.last_name}")
        with col_h2:
            if st.button("✏️ Edit", key=f"edit_person_{person.id}", use_container_width=True):
                edit_person_dialog(person)
        
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
                    "Source": a.source or "",
                    "User": a.created_by or "SYSTEM"
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
