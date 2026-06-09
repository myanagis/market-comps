from market_comps.db.session import SessionLocal
from market_comps.db.models import Organization, InvestorProfile
from sqlalchemy.exc import IntegrityError

db = SessionLocal()
orgs = db.query(Organization).filter(Organization.organization_type.is_(None)).all()
count = 0

for org in orgs:
    org.organization_type = 'INVESTOR'
    count += 1
    
    prof = db.query(InvestorProfile).filter_by(organization_id=org.id).first()
    if not prof:
        try:
            prof = InvestorProfile(organization_id=org.id, investor_type='Fund Manager')
            db.add(prof)
            db.flush()
        except IntegrityError:
            db.rollback()

db.commit()
print(f'Fixed {count} orgs.')
