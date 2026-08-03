from market_comps.db.session import engine
from sqlalchemy import text
with engine.connect() as conn:
    conn.execute(text('ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_total_tokens INTEGER DEFAULT 0'))
    conn.execute(text('ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_estimated_cost_usd FLOAT DEFAULT 0.0'))
    conn.execute(text('ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS exa_calls INTEGER DEFAULT 0'))
    conn.execute(text('ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS exa_estimated_cost_usd FLOAT DEFAULT 0.0'))
    conn.execute(text("UPDATE alembic_version SET version_num = '7b452db24615'"))
    conn.commit()
    print('DB Altered!')
