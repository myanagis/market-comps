import re
from pathlib import Path
from sqlalchemy import create_engine, text

secrets_path = Path(".streamlit/secrets.toml")
content = secrets_path.read_text(encoding="utf-8")
match = re.search(r'SUPABASE_DIRECT_URL\s*=\s*"([^"]+)"', content)
direct_url = match.group(1)

engine = create_engine(direct_url)
with engine.connect() as conn:
    print("Altering DB directly...")
    try:
        conn.execute(text('ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_total_tokens INTEGER DEFAULT 0'))
        conn.execute(text('ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_estimated_cost_usd FLOAT DEFAULT 0.0'))
        conn.execute(text('ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS exa_calls INTEGER DEFAULT 0'))
        conn.execute(text('ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS exa_estimated_cost_usd FLOAT DEFAULT 0.0'))
        conn.execute(text("UPDATE alembic_version SET version_num = '7b452db24615'"))
        conn.commit()
        print('DB Altered!')
    except Exception as e:
        print("Failed to alter table, likely columns exist or another error:", e)
        conn.rollback()
