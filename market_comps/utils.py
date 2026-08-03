import zoneinfo
import re
import re

def normalize_company_name(name: str) -> str:
    if not name:
        return ""
    # Lowercase and remove common suffixes like Inc, Corp, LLC, etc.
    name = name.lower().strip()
    name = re.sub(r'\b(inc|corp|corporation|llc|ltd|limited|co|company)\b\.?', '', name)
    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def format_est_datetime(dt):
    if not dt:
        return "Unknown Time"
    try:
        eastern = zoneinfo.ZoneInfo("America/New_York")
        tz_time = dt.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(eastern)
        return tz_time.strftime('%Y-%m-%d %I:%M %p ET')
    except Exception:
        return str(dt)

def format_currency(amount_str):
    if not amount_str:
        return ""
    
    # Try to extract the number
    amount_str = str(amount_str).strip()
    
    # If it already looks formatted like "$5M", just return it
    if "$" in amount_str and amount_str[-1].upper() in ("M", "B", "K"):
        return amount_str
        
    try:
        # Remove non-numeric characters except dot
        clean_str = re.sub(r'[^\d.]', '', amount_str)
        if clean_str:
            val = float(clean_str)
            if val.is_integer():
                return f"${int(val):,}"
            else:
                return f"${val:,.2f}"
    except Exception:
        pass
        
    return amount_str

def format_audit_row(a, db):
    source_str = a.source or ""
    if source_str == "PIPELINE" and a.extraction_job and a.extraction_job.pipeline_run:
        docs = a.extraction_job.pipeline_run.source_documents
        if docs and docs[0].document_date:
            source_str += f" (Doc: {docs[0].document_date})"

    raw_action = a.mutation_type
    if raw_action == "CREATE": action_str = "Add"
    elif raw_action == "UPDATE": action_str = "Update"
    elif raw_action in ["DELETE", "REMOVE"]: action_str = "Remove"
    else: action_str = raw_action
    
    old_val = a.old_value or ""
    new_val = a.new_value or ""
    
    field_str = a.field_name or a.canonical_entity_type or ""
    if field_str == "ORGANIZATION": field_str = "Company"
    elif field_str == "INVESTOR_PROFILE": field_str = "Investor Profile"
    elif field_str == "FUND_PROFILE": field_str = "Fund"
    elif field_str == "COMPANY_PROFILE": field_str = "Company Profile"
    elif field_str == "PERSON_ROLE": field_str = "Person Role"
    elif field_str == "PERSON": field_str = "Person"
    elif field_str == "PERSON_EMAIL": field_str = "Email"
    
    if raw_action == "CREATE" and not new_val:
        from market_comps.db.models import Person, PersonOrganizationRole, Organization
        if a.canonical_entity_type == "PERSON":
            p = db.query(Person).filter_by(id=a.canonical_entity_id).first()
            if p: new_val = p.full_name
        elif a.canonical_entity_type == "PERSON_ROLE":
            r = db.query(PersonOrganizationRole).filter_by(id=a.canonical_entity_id).first()
            if r and r.person: new_val = f"{r.person.full_name} ({r.title})"
        elif a.canonical_entity_type == "ORGANIZATION":
            o = db.query(Organization).filter_by(id=a.canonical_entity_id).first()
            if o: new_val = o.name
            
    val_str = ""
    if action_str == "Add":
        val_str = new_val
    elif action_str == "Update":
        val_str = f"{old_val} -> {new_val}"
    elif action_str == "Remove":
        val_str = f"old value: {old_val}"
        
    return {
        "Date": format_est_datetime(a.created_at),
        "Field": field_str,
        "Action": action_str,
        "Value": val_str,
        "Source": source_str,
        "User": a.created_by or "SYSTEM"
    }
