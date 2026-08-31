"""Case-history CRUD business logic, kept out of route handlers."""

from sqlalchemy.orm import Session

from backend.app.models.case import Case


def create_case(
    db: Session, image_path: str, predicted_class: str, probabilities: dict[str, float]
) -> Case:
    case = Case(
        image_path=image_path,
        predicted_class=predicted_class,
        probabilities=probabilities,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case
