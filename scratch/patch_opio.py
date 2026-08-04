from market_comps.db.session import SessionLocal
from sqlalchemy import text
db = SessionLocal()
db.execute(text("UPDATE organizations SET organization_type = 'COMPANY' WHERE organization_type IS NULL"))
db.commit()
