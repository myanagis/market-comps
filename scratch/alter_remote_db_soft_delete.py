import re
from pathlib import Path
from sqlalchemy import create_engine, text

secrets_path = Path(".streamlit/secrets.toml")
content = secrets_path.read_text(encoding="utf-8")
match = re.search(r'SUPABASE_DIRECT_URL\s*=\s*"([^"]+)"', content)
direct_url = match.group(1)

engine = create_engine(direct_url)
with engine.connect() as conn:
    print("Terminating other connections...")
    conn.execute(text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = current_database() AND usename = current_user;"))
    conn.commit()

with engine.connect() as conn:
    print("Altering DB directly...")
    try:
        conn.execute(text('ALTER TABLE source_documents ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP DEFAULT NULL'))
        conn.execute(text('ALTER TABLE source_documents ADD COLUMN IF NOT EXISTS deleted_by VARCHAR DEFAULT NULL'))
        conn.execute(text('ALTER TABLE investments ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP DEFAULT NULL'))
        conn.execute(text('ALTER TABLE investments ADD COLUMN IF NOT EXISTS deleted_by VARCHAR DEFAULT NULL'))
        conn.commit()
        print('DB Altered!')
    except Exception as e:
        print("Failed to alter table:", e)
        conn.rollback()
