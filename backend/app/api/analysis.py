from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.app.agents.orchestrator import (
    dermatology_orchestrator,
)

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    # ---------------------------------------------
    # 1. Validate uploaded file
    # ---------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported.",
        )

    # ---------------------------------------------
    # 2. Create unique filename
    # ---------------------------------------------

    original_name = Path(
        file.filename or "uploaded_image"
    ).stem

    unique_id = uuid4().hex[:8]

    input_filename = (
        f"{original_name}_{unique_id}.png"
    )

    file_path = UPLOAD_DIR / input_filename

    # ---------------------------------------------
    # 3. Save uploaded image
    # ---------------------------------------------

    try:

        file_bytes = await file.read()

        with open(file_path, "wb") as buffer:
            buffer.write(file_bytes)

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded image: {error}",
        )

    # ---------------------------------------------
    # 4. Run Dermatology Agent Orchestrator
    # ---------------------------------------------

    try:

        result = dermatology_orchestrator.analyze(
            str(file_path)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Dermatology Agent Orchestrator failed: {error}",
        )

    # ---------------------------------------------
    # 5. Return combined agent result
    # ---------------------------------------------

    return {
        **result,

        "filename": file.filename,

        "uploaded_file": input_filename,
    }