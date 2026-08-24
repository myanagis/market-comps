from market_comps.db.session import engine
from sqlalchemy import text

def migrate():
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE markets ADD COLUMN sectors JSON;"))
            print("Added sectors to markets")
        except Exception as e:
            print("markets.sectors:", e)
        conn.commit()

if __name__ == "__main__":
    migrate()
