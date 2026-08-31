"""Case-history CRUD business logic, kept out of route handlers."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.case import Case


def create_case(
    db: Session,
    image_path: str,
    gradcam_image_path: str,
    predicted_class: str,
    probabilities: dict[str, float],
) -> Case:
    case = Case(
        image_path=image_path,
        gradcam_image_path=gradcam_image_path,
        predicted_class=predicted_class,
        probabilities=probabilities,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def list_cases(db: Session, limit: int = 50, offset: int = 0) -> list[Case]:
    """Most recent cases first, for the GET /cases history view."""
    stmt = (
        select(Case)
        .order_by(Case.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


def get_case(db: Session, case_id: uuid.UUID) -> Case | None:
    return db.get(Case, case_id)


def delete_case(db: Session, case_id: uuid.UUID) -> bool:
    """Deletes the DB row only; the stored image file is left on disk.

    File cleanup deferred along with save_gradcam_overlay() -- see
    app/storage/files.py -- until storage has a real retention story.
    Returns False if no such case exists.
    """
    case = db.get(Case, case_id)
    if case is None:
        return False
    db.delete(case)
    db.commit()
    return True
