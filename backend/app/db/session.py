"""SQLAlchemy engine, SessionLocal, and the get_db() request-scoped session dependency."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    # One session per request: opened here, closed in the finally block
    # regardless of whether the request handler raised.
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
