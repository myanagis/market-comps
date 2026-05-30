import psycopg2
from market_comps.config import settings

def kill_idle_connections():
    print(f"Connecting to {settings.DATABASE_URL.replace(settings.DATABASE_PASSWORD, '***')}")
    conn = psycopg2.connect(settings.DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()
    # Try to terminate all other connections
    try:
        cur.execute("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE pid <> pg_backend_pid()
          AND state in ('idle', 'idle in transaction', 'active');
        """)
        print(f"Terminated {cur.rowcount} connections.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    kill_idle_connections()
