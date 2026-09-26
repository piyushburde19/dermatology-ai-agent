from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from backend.app.ml.predict import (
    model,
    TRANSFORM,
    DEVICE,
    CLASS_NAMES,
)


GRADCAM_DIR = Path("models/gradcam")
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


def generate_gradcam(
    image_path: str,
    output_filename: str,
    predicted_class: str,
):
    """
    Generate a Grad-CAM visualization for an uploaded image.

    Uses the same global model and preprocessing pipeline
    as predict.py so that prediction and explanation stay consistent.
    """

    image_path = Path(image_path)
    output_path = GRADCAM_DIR / output_filename

    # Make sure predicted class exists
    if predicted_class not in CLASS_NAMES:
        raise ValueError(
            f"Unknown predicted class: {predicted_class}"
        )

    class_index = CLASS_NAMES.index(predicted_class)

    # Load original image
    original_image = Image.open(image_path).convert("RGB")

    # Prepare model input using the SAME transform as prediction
    input_tensor = TRANSFORM(original_image).unsqueeze(0).to(DEVICE)

    # EfficientNet-B0 final convolutional feature layer
    target_layers = [
        model.model.features[-1]
    ]

    model.eval()

    # Grad-CAM needs gradients, so explicitly enable them
    with torch.enable_grad():

        targets = [
            ClassifierOutputTarget(class_index)
        ]

        with GradCAM(
            model=model,
            target_layers=target_layers,
        ) as cam:

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
            )

            grayscale_cam = grayscale_cam[0]

    # Resize original image to match model input
    visualization_image = original_image.resize(
        (224, 224)
    )

    rgb_image = np.asarray(
        visualization_image
    ).astype(np.float32) / 255.0

    # Generate colored Grad-CAM overlay
    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
    )

    # Save result
    Image.fromarray(visualization).save(
        output_path,
        quality=95,
    )

    return {
        "filename": output_filename,
        "path": str(output_path),
        "url": f"/gradcam/{output_filename}",
        "predicted_class": predicted_class,
    }