from market_comps.db.session import SessionLocal
from market_comps.db.models import Transaction

db = SessionLocal()
for t in db.query(Transaction).all():
    print(f"Target: {t.target_company_id}, Acquirer: {t.acquirer_company_id}, Type: {t.transaction_type}, Date: {t.announced_date}, Value: {t.transaction_value_numeric}")
