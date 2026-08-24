from market_comps.db.session import engine
from sqlalchemy import text
from market_comps.db.models import Base

def migrate():
    # Create any new tables (e.g. Sector, ComparisonSetOrganizationLink)
    Base.metadata.create_all(bind=engine)
    
    with engine.connect() as conn:
        # Add sectors to CompanyProfile
        try:
            conn.execute(text("ALTER TABLE company_profiles ADD COLUMN sectors JSON;"))
            print("Added sectors to company_profiles")
        except Exception as e:
            print("company_profiles.sectors:", e)
            
        # Add new fields to InvestorProfile
        try:
            conn.execute(text("ALTER TABLE investor_profiles ADD COLUMN sectors JSON;"))
            print("Added sectors to investor_profiles")
        except Exception as e:
            print("investor_profiles.sectors:", e)

        try:
            conn.execute(text("ALTER TABLE investor_profiles ADD COLUMN stages JSON;"))
            print("Added stages to investor_profiles")
        except Exception as e:
            print("investor_profiles.stages:", e)

        try:
            conn.execute(text("ALTER TABLE investor_profiles ADD COLUMN specialties JSON;"))
            print("Added specialties to investor_profiles")
        except Exception as e:
            print("investor_profiles.specialties:", e)

        try:
            conn.execute(text("ALTER TABLE investor_profiles ADD COLUMN check_size_min FLOAT;"))
            print("Added check_size_min to investor_profiles")
        except Exception as e:
            print("investor_profiles.check_size_min:", e)

        try:
            conn.execute(text("ALTER TABLE investor_profiles ADD COLUMN check_size_max FLOAT;"))
            print("Added check_size_max to investor_profiles")
        except Exception as e:
            print("investor_profiles.check_size_max:", e)

        # Add sector to Market
        try:
            conn.execute(text("ALTER TABLE markets ADD COLUMN sector VARCHAR;"))
            print("Added sector to markets")
        except Exception as e:
            print("markets.sector:", e)

        conn.commit()

if __name__ == "__main__":
    migrate()
