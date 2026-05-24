import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import text
from market_comps.db.session import engine

def main():
    try:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE pipeline_runs CASCADE;"))
            print("Truncated tables.")
            
    except Exception as e:
        print(f"Error truncating tables: {e}")

if __name__ == "__main__":
    main()
