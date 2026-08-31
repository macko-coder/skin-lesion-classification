"""Case-history endpoints: list/get/delete past predictions."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db
from backend.app.schemas.case import CaseListItem, CaseRead
from backend.app.services.case_service import delete_case, get_case, list_cases

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("", response_model=list[CaseListItem])
def read_cases(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[CaseListItem]:
    return list_cases(db, limit=limit, offset=offset)


@router.get("/{case_id}", response_model=CaseRead)
def read_case(case_id: uuid.UUID, db: Session = Depends(get_db)) -> CaseRead:
    case = get_case(db, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.delete("/{case_id}", status_code=204)
def remove_case(case_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    if not delete_case(db, case_id):
        raise HTTPException(status_code=404, detail="Case not found")
