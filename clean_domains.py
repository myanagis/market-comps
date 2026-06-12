from sqlalchemy.orm import Session
from market_comps.db.session import SessionLocal
from market_comps.db.models import Organization

def clean_domain(value):
    if not value:
        return value
        
    value = value.lower().strip()
    if "://" in value:
        value = value.split("://")[-1]
    if "/" in value:
        value = value.split("/")[0]
    if value.startswith("www."):
        value = value[4:]
        
    return value if value else None

def clean_all_domains():
    db = SessionLocal()
    orgs = db.query(Organization).all()
    cleaned = 0
    for org in orgs:
        if org.primary_domain:
            new_domain = clean_domain(org.primary_domain)
            if new_domain != org.primary_domain:
                print(f"Cleaning: {org.primary_domain} -> {new_domain}")
                org.primary_domain = new_domain
                cleaned += 1
                
    if cleaned > 0:
        try:
            db.commit()
            print(f"Successfully cleaned {cleaned} domains.")
        except Exception as e:
            db.rollback()
            print(f"Error committing: {e}")
    else:
        print("No domains needed cleaning.")
    db.close()

if __name__ == "__main__":
    clean_all_domains()
