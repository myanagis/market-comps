import sqlalchemy
from market_comps.config import settings

def kill_connections():
    if not settings.supabase_api_url:
        return
    
    # We need the direct database URL, not the API URL
    import os
    db_url = os.environ.get("SUPABASE_DIRECT_URL")
    if not db_url:
        import streamlit as st
        try:
            db_url = st.secrets.get("SUPABASE_DIRECT_URL")
        except:
            pass
            
    if not db_url:
        print("No DB URL")
        return

    engine = sqlalchemy.create_engine(db_url)
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        res = conn.execute(sqlalchemy.text("""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE pid <> pg_backend_pid()
            AND state = 'idle in transaction';
        """))
        print(f"Terminated some connections: {res.rowcount if hasattr(res, 'rowcount') else 'unknown'}")

if __name__ == "__main__":
    kill_connections()
