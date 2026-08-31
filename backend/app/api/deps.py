"""Shared FastAPI dependencies: get_db() session, ML inference singleton access."""

from backend.app.db.session import get_db

__all__ = ["get_db"]
