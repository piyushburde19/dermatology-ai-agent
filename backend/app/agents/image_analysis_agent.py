from pathlib import Path

from backend.app.ml.predict import predict_skin_disease
from backend.app.ml.gradcam_service import generate_gradcam


class ImageAnalysisAgent:
    """
    Image Analysis Agent

    Responsibilities:
    - Run dermatology image classification
    - Return prediction probabilities
    - Return confidence
    - Return OOD-related metrics
    - Generate Grad-CAM when the image is supported
    """

    name = "Image Analysis Agent"

    def analyze(self, image_path: str):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # ---------------------------------------------
        # 1. Run existing ML prediction pipeline
        # ---------------------------------------------

        result = predict_skin_disease(
            str(image_path)
        )

        # ---------------------------------------------
        # 2. Generate Grad-CAM for supported images
        # ---------------------------------------------

        gradcam_url = None
        gradcam_available = False

        if not result.get("ood", False):

            try:
                gradcam_filename = (
                    f"agent_gradcam_{image_path.stem}.jpg"
                )

                gradcam_result = generate_gradcam(
                    image_path=str(image_path),
                    output_filename=gradcam_filename,
                    predicted_class=result["prediction"],
                )

                gradcam_url = gradcam_result["url"]
                gradcam_available = True

            except Exception as error:
                print(
                    f"[Image Analysis Agent] "
                    f"Grad-CAM failed: {error}"
                )

        # ---------------------------------------------
        # 3. Return structured agent result
        # ---------------------------------------------

        return {
            "agent": self.name,

            "prediction": result.get(
                "prediction"
            ),

            "raw_prediction": result.get(
                "raw_prediction"
            ),

            "confidence": result.get(
                "confidence"
            ),

            "probabilities": result.get(
                "probabilities",
                {}
            ),

            "similarity": result.get(
                "similarity"
            ),

            "entropy": result.get(
                "entropy"
            ),

            "ood": result.get(
                "ood",
                False
            ),

            "ood_thresholds": result.get(
                "ood_thresholds",
                {}
            ),

            "gradcam_available":
                gradcam_available,

            "gradcam_url":
                gradcam_url,

            "status": "completed",
        }


# --------------------------------------------------
# Shared agent instance
# --------------------------------------------------

image_analysis_agent = ImageAnalysisAgent()