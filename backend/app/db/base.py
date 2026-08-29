"""SQLAlchemy declarative Base; imports app.models.case so Alembic autogenerate can see it."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Imported for its side effect: registers Case on Base.metadata so Alembic's
# autogenerate (via alembic/env.py) can detect it. Import goes at the bottom
# to avoid a circular import (case.py imports Base from this module).
from backend.app.models.case import Case  # noqa: E402,F401
