"""SQLAlchemy declarative Base.

Deliberately does NOT import concrete models (e.g. Case) here -- that
previously caused a real circular import (case.py imports Base from this
module; this module importing case.py back created a cycle that broke
depending on which module happened to be imported first). Whatever needs
every model registered on Base.metadata (Alembic autogenerate, or a test
setup calling Base.metadata.create_all()) imports the model modules
explicitly itself -- see alembic/env.py.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
