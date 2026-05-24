import sys
import os

# Add the parent directory to sys.path so we can import market_comps
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import text
from market_comps.db.session import engine

def main():
    try:
        with engine.connect() as conn:
            # Terminate other connections to release locks
            conn.execute(text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND state = 'idle';"))
            conn.commit()
            print("Terminated idle connections.")
            
    except Exception as e:
        print(f"Error terminating connections: {e}")

if __name__ == "__main__":
    main()
