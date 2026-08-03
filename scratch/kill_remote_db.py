from sqlalchemy import create_engine, text
import re
from pathlib import Path

secrets_path = Path(".streamlit/secrets.toml")
content = secrets_path.read_text(encoding="utf-8")
match = re.search(r'SUPABASE_DIRECT_URL\s*=\s*"([^"]+)"', content)
direct_url = match.group(1)

engine = create_engine(direct_url)
with engine.connect() as conn:
    print("Killing connections...")
    conn.execute(text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = 'postgres' AND usename = 'postgres'"))
    conn.commit()
    print("Done")
