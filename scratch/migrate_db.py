import sqlite3

def migrate():
    conn = sqlite3.connect('market_comps.db')
    cursor = conn.cursor()
    
    # 1. Add sectors to CompanyProfile
    try:
        cursor.execute("ALTER TABLE company_profiles ADD COLUMN sectors JSON;")
        print("Added sectors to company_profiles")
    except Exception as e:
        print("company_profiles.sectors:", e)
        
    # 2. Add new fields to InvestorProfile
    try:
        cursor.execute("ALTER TABLE investor_profiles ADD COLUMN sectors JSON;")
        print("Added sectors to investor_profiles")
    except Exception as e:
        print("investor_profiles.sectors:", e)

    try:
        cursor.execute("ALTER TABLE investor_profiles ADD COLUMN stages JSON;")
        print("Added stages to investor_profiles")
    except Exception as e:
        print("investor_profiles.stages:", e)

    try:
        cursor.execute("ALTER TABLE investor_profiles ADD COLUMN specialties JSON;")
        print("Added specialties to investor_profiles")
    except Exception as e:
        print("investor_profiles.specialties:", e)

    try:
        cursor.execute("ALTER TABLE investor_profiles ADD COLUMN check_size_min FLOAT;")
        print("Added check_size_min to investor_profiles")
    except Exception as e:
        print("investor_profiles.check_size_min:", e)

    try:
        cursor.execute("ALTER TABLE investor_profiles ADD COLUMN check_size_max FLOAT;")
        print("Added check_size_max to investor_profiles")
    except Exception as e:
        print("investor_profiles.check_size_max:", e)

    # 3. Add sector to Market
    try:
        cursor.execute("ALTER TABLE markets ADD COLUMN sector VARCHAR;")
        print("Added sector to markets")
    except Exception as e:
        print("markets.sector:", e)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    migrate()
