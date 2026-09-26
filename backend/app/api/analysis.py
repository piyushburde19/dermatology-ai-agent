from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.app.ml.predict import predict_skin_disease
from backend.app.ml.gradcam_service import generate_gradcam


router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
GRADCAM_DIR = Path("models/gradcam")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...)
):
    # --------------------------------------------------
    # 1. Validate file type
    # --------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    # --------------------------------------------------
    # 2. Create safe unique filename
    # --------------------------------------------------

    original_name = Path(
        file.filename or "uploaded_image"
    ).stem

    unique_id = uuid4().hex[:8]

    input_filename = (
        f"{original_name}_{unique_id}.png"
    )

    file_path = UPLOAD_DIR / input_filename

    # --------------------------------------------------
    # 3. Save uploaded image
    # --------------------------------------------------

    try:
        file_bytes = await file.read()

        with open(file_path, "wb") as buffer:
            buffer.write(file_bytes)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded image: {str(e)}"
        )

    # --------------------------------------------------
    # 4. Run prediction + OOD analysis
    # --------------------------------------------------

    try:
        result = predict_skin_disease(
            str(file_path)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # --------------------------------------------------
    # 5. Generate Grad-CAM
    # --------------------------------------------------

    gradcam_url = None
    gradcam_available = False

    # We only show Grad-CAM when the image passes
    # the current OOD screening.
    if not result.get("ood", False):

        try:
            gradcam_filename = (
                f"gradcam_{original_name}_{unique_id}.jpg"
            )

            gradcam_result = generate_gradcam(
                image_path=str(file_path),
                output_filename=gradcam_filename,
                predicted_class=result["prediction"],
            )

            gradcam_url = gradcam_result["url"]
            gradcam_available = True

        except Exception as e:
            # Do not fail the whole prediction if
            # explanation generation fails.
            print(
                f"Grad-CAM generation failed: {e}"
            )

    # --------------------------------------------------
    # 6. Return complete analysis result
    # --------------------------------------------------

    return {
        **result,

        "filename": file.filename,

        "gradcam_available": gradcam_available,

        "gradcam_url": gradcam_url,
    }