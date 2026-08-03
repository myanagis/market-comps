from market_comps.db.session import engine
from sqlalchemy import text
try:
    with engine.connect() as conn:
        conn.execute(text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid != pg_backend_pid() AND datname = 'postgres'"))
        conn.commit()
except Exception as e:
    print(e)
