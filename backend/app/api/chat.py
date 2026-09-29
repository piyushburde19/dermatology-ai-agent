from typing import Optional, Dict

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


CLASS_NAMES = {
    "ACK": "Actinic Keratosis",
    "BCC": "Basal Cell Carcinoma",
    "MEL": "Melanoma",
    "NEV": "Nevus",
    "SCC": "Squamous Cell Carcinoma",
    "SEK": "Seborrheic Keratosis",
}


class ChatRequest(BaseModel):
    message: str

    prediction: Optional[str] = None
    confidence: Optional[float] = None
    ood: Optional[bool] = None

    similarity: Optional[float] = None
    entropy: Optional[float] = None
    assessment: Optional[str] = None
    review_required: Optional[bool] = None
    gradcam_available: Optional[bool] = None
    probabilities: Optional[Dict[str, float]] = None


def format_class_name(class_code: Optional[str]) -> str:

    if not class_code:
        return "the predicted class"

    return CLASS_NAMES.get(
        class_code,
        class_code,
    )


def get_prediction_context(request: ChatRequest):

    if request.ood:
        return (
            "The uploaded image was marked as "
            "Unsupported / Uncertain by the system."
        )

    if not request.prediction:
        return (
            "No image analysis result is currently "
            "available."
        )

    condition = format_class_name(
        request.prediction
    )

    confidence_text = ""

    if request.confidence is not None:
        confidence_text = (
            f" with {request.confidence:.2f}% confidence"
        )

    return (
        f"The current AI analysis predicts "
        f"{condition}{confidence_text}."
    )


@router.post("/chat")
async def chat(request: ChatRequest):

    message = request.message.strip().lower()

    prediction = request.prediction
    condition = format_class_name(prediction)

    # ==========================================================
    # NO ANALYSIS AVAILABLE
    # ==========================================================

    if not prediction and not request.ood:

        return {
            "response": (
                "Please upload and analyze a dermatology "
                "image first. Once an analysis is available, "
                "I can explain the prediction, confidence, "
                "uncertainty signals, and next steps."
            )
        }

    # ==========================================================
    # OUT-OF-DOMAIN / UNSUPPORTED IMAGE
    # ==========================================================

    if request.ood:

        if (
            "why" in message
            or "result" in message
            or "mean" in message
        ):

            return {
                "response": (
                    "The system marked this image as "
                    "Unsupported / Uncertain because it did "
                    "not meet the model's supported-image "
                    "criteria. The model may still produce "
                    "a raw prediction internally, but that "
                    "prediction should not be treated as a "
                    "reliable classification for this image."
                )
            }

        if (
            "next" in message
            or "do" in message
            or "should" in message
        ):

            return {
                "response": (
                    "Because the image was marked "
                    "Unsupported / Uncertain, the system "
                    "should not be relied on for a disease "
                    "classification from this image. "
                    "Consider obtaining an appropriate "
                    "clinical image and, if the lesion is "
                    "concerning, seek evaluation by a "
                    "qualified dermatologist."
                )
            }

        return {
            "response": (
                "This image was marked "
                "Unsupported / Uncertain by the AI system. "
                "I can explain the uncertainty result and "
                "what the system's analysis means."
            )
        }

    # ==========================================================
    # WHAT DOES MY RESULT MEAN?
    # ==========================================================

    if (
        "what does" in message
        or "what is my result" in message
        or "result mean" in message
        or "meaning" in message
    ):

        confidence_text = ""

        if request.confidence is not None:
            confidence_text = (
                f" The model reported "
                f"{request.confidence:.2f}% confidence."
            )

        return {
            "response": (
                f"The image-analysis system predicted "
                f"{condition}.{confidence_text} "
                f"This is an AI-assisted preliminary "
                f"prediction and does not confirm that the "
                f"condition is medically present."
            )
        }

    # ==========================================================
    # WHY DID THE SYSTEM GIVE THIS RESULT?
    # ==========================================================

    if (
        "why" in message
        and (
            "result" in message
            or "prediction" in message
            or "give" in message
            or "this" in message
        )
    ):

        confidence_text = ""

        if request.confidence is not None:
            confidence_text = (
                f"The reported confidence was "
                f"{request.confidence:.2f}%."
            )

        uncertainty_text = ""

        if request.assessment == "Review Recommended":
            uncertainty_text = (
                " The uncertainty agent also recommended "
                "human review of this result."
            )
        elif request.assessment == "Supported":
            uncertainty_text = (
                " The image passed the system's supported "
                "image checks."
            )

        probability_text = ""

        if request.probabilities:

            sorted_probs = sorted(
                request.probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            top_classes = sorted_probs[:2]

            if top_classes:

                probability_text = (
                    " The model's class probabilities "
                    "were used to select the highest-"
                    "probability class."
                )

        return {
            "response": (
                f"The system predicted {condition} because "
                f"this was the model's highest-probability "
                f"supported class for the uploaded image."
                f"{probability_text}"
                f"{confidence_text}"
                f"{uncertainty_text} "
                "The prediction should be interpreted as "
                "AI-assisted analysis rather than a confirmed "
                "medical diagnosis."
            )
        }

    # ==========================================================
    # CONFIDENCE
    # ==========================================================

    if (
        "confidence" in message
        or "sure" in message
        or "certain" in message
    ):

        if request.confidence is not None:

            return {
                "response": (
                    f"The model reported "
                    f"{request.confidence:.2f}% confidence "
                    f"for the predicted class, "
                    f"{condition}. Confidence is a model "
                    f"output and should not be interpreted "
                    f"as the probability that a patient "
                    f"actually has the condition."
                )
            }

        return {
            "response": (
                "The confidence value is not available "
                "for this analysis."
            )
        }

    # ==========================================================
    # UNCERTAINTY / REVIEW
    # ==========================================================

    if (
        "uncertain" in message
        or "uncertainty" in message
        or "review" in message
        or "risk" in message
    ):

        if request.review_required:

            return {
                "response": (
                    "The Risk & Uncertainty Agent has "
                    "recommended human review. This can "
                    "happen when the prediction confidence "
                    "is below the configured threshold or "
                    "when prediction uncertainty is elevated."
                )
            }

        return {
            "response": (
                "The current analysis was not routed for "
                "additional review by the Risk & Uncertainty "
                "Agent."
            )
        }

    # ==========================================================
    # GRAD-CAM / EXPLANATION
    # ==========================================================

    if (
        "grad-cam" in message
        or "gradcam" in message
        or "explain" in message
        or "look" in message
        or "attention" in message
    ):

        if request.gradcam_available:

            return {
                "response": (
                    "Grad-CAM provides a visual explanation "
                    "of the image regions that contributed "
                    "to the model's prediction. It helps "
                    "inspect where the model focused, but "
                    "the heatmap is not proof that those "
                    "regions establish a medical diagnosis."
                )
            }

        return {
            "response": (
                "A Grad-CAM explanation is not available "
                "for this analysis."
            )
        }

    # ==========================================================
    # NEXT STEP
    # ==========================================================

    if (
        "what should i do" in message
        or "what should i do next" in message
        or "next step" in message
        or "next steps" in message
        or "what do i do" in message
    ):

        return {
            "response": (
                f"The AI system predicted {condition}, "
                "but this should not be treated as a "
                "confirmed diagnosis. If the skin lesion "
                "is persistent, changing, painful, bleeding, "
                "growing, or otherwise concerning, consider "
                "evaluation by a qualified dermatologist."
            )
        }

    # ==========================================================
    # OUT-OF-CONTEXT QUESTIONS
    # ==========================================================

    weather_keywords = [
        "weather",
        "temperature",
        "forecast",
        "rain",
        "snow",
    ]

    if any(
        keyword in message
        for keyword in weather_keywords
    ):

        return {
            "response": (
                "I'm the Dermatology AI Assistant, so I'm "
                "designed to answer questions related to "
                "the uploaded skin image and its analysis. "
                "I can't provide weather information."
            )
        }

    # ==========================================================
    # DEFAULT DERMATOLOGY RESPONSE
    # ==========================================================

    return {
        "response": (
            f"The current AI analysis predicts {condition}. "
            "I can explain the prediction, confidence, "
            "uncertainty signals, Grad-CAM explanation, "
            "or possible next steps."
        )
    }