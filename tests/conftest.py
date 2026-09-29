import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from market_comps.db.models import Base
from market_comps.db.session import engine as real_engine

@pytest.fixture(scope="session")
def test_engine():
    # Use SQLite in-memory for fast testing, or you can use a test Postgres DB
    # Note: SQLite has some dialect differences with Postgres (like JSONB), 
    # but works for basic model testing. For production, consider testcontainers.
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def test_db(test_engine):
    """Returns a sqlalchemy session, and rolls back any changes after the test."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    yield session
    session.rollback()
    session.close()
