import os
import re
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

def get_database_url(direct: bool = False) -> str:
    """Read DB URL from st.secrets or fallback to simple parsing of secrets.toml for Alembic."""
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_DIRECT_URL" if direct else "SUPABASE_URL")
        if url:
            return url
    except Exception:
        pass
    
    # Fallback to local regex parsing if running outside streamlit (like alembic)
    secrets_path = Path(__file__).resolve().parent.parent.parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        content = secrets_path.read_text(encoding="utf-8")
        key = "SUPABASE_DIRECT_URL" if direct else "SUPABASE_URL"
        match = re.search(fr'{key}\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
            
    return os.environ.get("SUPABASE_DIRECT_URL" if direct else "SUPABASE_URL", "")

# We use the standard URL for the app engine
url = get_database_url(direct=False)
if not url:
    url = "sqlite:///:memory:"

engine = create_engine(url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def _get_active_session_maker():
    global engine, SessionLocal
    # If the engine was initialized with sqlite memory before secrets were loaded,
    # try to re-initialize it now.
    if str(engine.url) == "sqlite:///:memory:":
        new_url = get_database_url(direct=False)
        if new_url and new_url != "sqlite:///:memory:":
            engine = create_engine(new_url, pool_pre_ping=True)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal

def get_db():
    session_factory = _get_active_session_maker()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_context():
    session_factory = _get_active_session_maker()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
