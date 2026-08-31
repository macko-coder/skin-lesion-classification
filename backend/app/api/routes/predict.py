"""POST endpoint: uploaded lesion image -> inference (+ Grad-CAM), persists a Case."""

import io

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db
from backend.app.ml.inference import predict as run_inference
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.case_service import create_case
from backend.app.storage.files import save_upload

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile, db: Session = Depends(get_db)) -> PredictionResponse:
    file_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # force full decode now, so a corrupt/non-image upload fails here with a 400
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    probabilities = run_inference(image)
    predicted_class = max(probabilities, key=probabilities.get)

    image_path = save_upload(file_bytes, file.filename or "upload.jpg")
    create_case(
        db,
        image_path=image_path,
        predicted_class=predicted_class,
        probabilities=probabilities,
    )

    return PredictionResponse(predicted_class=predicted_class, probabilities=probabilities)
