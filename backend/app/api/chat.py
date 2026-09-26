from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str
    prediction: str | None = None
    confidence: float | None = None
    ood: bool | None = None


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_INFO = {
    "ACK": {
        "name": "Actinic Keratosis",
        "description": (
            "Actinic keratosis is a rough or scaly skin lesion "
            "that can occur on areas of skin exposed to sunlight."
        ),
    },

    "BCC": {
        "name": "Basal Cell Carcinoma",
        "description": (
            "Basal cell carcinoma is a type of skin cancer that "
            "commonly develops in sun-exposed areas of the skin."
        ),
    },

    "MEL": {
        "name": "Melanoma",
        "description": (
            "Melanoma is a type of skin cancer that develops "
            "from pigment-producing cells."
        ),
    },

    "NEV": {
        "name": "Nevus",
        "description": (
            "A nevus is a common type of mole or pigmented skin lesion."
        ),
    },

    "SCC": {
        "name": "Squamous Cell Carcinoma",
        "description": (
            "Squamous cell carcinoma is a type of skin cancer "
            "that can occur on different areas of the skin."
        ),
    },

    "SEK": {
        "name": "Seborrheic Keratosis",
        "description": (
            "Seborrheic keratosis is a common benign skin growth "
            "that can appear as a raised, rough, or waxy lesion."
        ),
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_class_info(prediction: str):
    """
    Convert the model class code into human-readable information.
    """

    if prediction is None:
        return None

    return CLASS_INFO.get(
        prediction.upper(),
        {
            "name": prediction,
            "description": (
                "The system returned a prediction that is not "
                "mapped to a known supported class."
            ),
        },
    )


def format_confidence(confidence: float | None) -> str:
    """
    Format confidence safely.
    """

    if confidence is None:
        return "an unspecified confidence"

    return f"{confidence:.2f}%"


# ============================================================
# RESPONSE GENERATOR
# ============================================================

def generate_response(request: ChatRequest) -> str:

    message = request.message.lower().strip()

    prediction = request.prediction
    confidence = request.confidence
    ood = request.ood

    class_info = get_class_info(prediction)

    class_name = (
        class_info["name"]
        if class_info
        else prediction
    )

    class_description = (
        class_info["description"]
        if class_info
        else ""
    )


    # ========================================================
    # GREETING
    # ========================================================

    if message in [
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
    ]:

        return (
            "Hello! I am the Dermatology AI Assistant. "
            "I can help explain the AI image-analysis result, "
            "confidence, uncertainty screening, supported classes, "
            "and general dermatology concepts."
        )


    # ========================================================
    # OOD / UNSUPPORTED IMAGE
    # ========================================================

    if ood:

        if any(word in message for word in [
            "result",
            "prediction",
            "mean",
            "why",
            "uncertain",
            "wrong",
            "ood",
            "unsupported",
        ]):

            return (
                "The uploaded image was marked as "
                "Unsupported / Uncertain by the image-analysis "
                "system. This means the image did not meet the "
                "system's current similarity and uncertainty "
                "criteria for the supported dermatology classes. "
                "The result should not be interpreted as a medical "
                "diagnosis."
            )


        if any(word in message for word in [
            "what",
            "explain",
            "tell",
        ]):

            return (
                "The system could not confidently place this image "
                "within its supported dermatology classes. "
                "This is an uncertainty screening result rather "
                "than a diagnosis. A qualified healthcare "
                "professional should evaluate any concerning "
                "skin finding."
            )


        if any(word in message for word in [
            "do",
            "next",
            "doctor",
            "dermatologist",
        ]):

            return (
                "Because the image was marked as "
                "Unsupported / Uncertain, the AI result should "
                "not be used to determine what the skin finding "
                "is. If the lesion is persistent, changing, "
                "painful, bleeding, or otherwise concerning, "
                "consider evaluation by a qualified dermatologist."
            )


        return (
            "The image-analysis system marked this image as "
            "Unsupported / Uncertain. I can explain the result "
            "and the uncertainty screening, but I cannot provide "
            "a medical diagnosis from the image."
        )


    # ========================================================
    # RESULT EXPLANATION
    # ========================================================

    if prediction:

        if any(phrase in message for phrase in [
            "what does my result mean",
            "what does the result mean",
            "what does my prediction mean",
            "what does this result mean",
            "explain my result",
            "explain the result",
            "explain prediction",
            "what is my result",
        ]):

            confidence_text = format_confidence(
                confidence
            )

            return (
                f"The image-analysis system predicted "
                f"{class_name} ({prediction}) with "
                f"{confidence_text} confidence. "
                f"{class_description} "
                f"The confidence represents the model's "
                f"estimated probability for its predicted class "
                f"within the supported classes. It does not mean "
                f"that the condition has been medically confirmed. "
                f"This is an AI-assisted preliminary analysis and "
                f"should not be treated as a medical diagnosis."
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        if any(word in message for word in [
            "confidence",
            "accurate",
            "accuracy",
            "sure",
            "certain",
        ]):

            confidence_text = format_confidence(
                confidence
            )

            return (
                f"The model assigned {confidence_text} confidence "
                f"to the predicted class, {class_name}. "
                f"This confidence is a model output and should "
                f"not be interpreted as the probability that a "
                f"doctor would make the same diagnosis. "
                f"Clinical examination is still required."
            )


        # ====================================================
        # WHAT IS THE PREDICTED CONDITION?
        # ====================================================

        if (
            "what is" in message
            or "what does" in message
            or "tell me about" in message
            or "explain" in message
        ) and (
            prediction.lower() in message
            or class_name.lower() in message
            or "condition" in message
            or "disease" in message
        ):

            return (
                f"The predicted class is {class_name} ({prediction}). "
                f"{class_description} "
                f"In this project, the model's prediction is used "
                f"for preliminary image analysis only and does not "
                f"confirm the presence of the condition."
            )


        # ====================================================
        # WHAT SHOULD I DO?
        # ====================================================

        if any(phrase in message for phrase in [
            "what should i do",
            "what should i do next",
            "what do i do",
            "next step",
            "next steps",
            "should i see a doctor",
            "should i see dermatologist",
        ]):

            return (
                f"The AI system predicted {class_name}, but this "
                f"should not be treated as a confirmed diagnosis. "
                f"If the skin lesion is persistent, changing, "
                f"painful, bleeding, growing, or otherwise "
                f"concerning, consider evaluation by a qualified "
                f"dermatologist."
            )


        # ====================================================
        # IS IT DANGEROUS?
        # ====================================================

        if any(phrase in message for phrase in [
            "is it dangerous",
            "is this dangerous",
            "should i worry",
            "is it serious",
            "is this serious",
        ]):

            return (
                f"The AI prediction of {class_name} alone cannot "
                f"determine whether a skin finding is dangerous "
                f"or serious. The model provides preliminary "
                f"image analysis only. A dermatologist should "
                f"evaluate concerning or changing lesions."
            )


        # ====================================================
        # GENERAL CONDITION QUESTION
        # ====================================================

        if (
            class_name
            and (
                class_name.lower() in message
                or prediction.lower() in message
            )
        ):

            return (
                f"{class_name} ({prediction}): "
                f"{class_description} "
                f"Remember that the AI prediction in this project "
                f"is preliminary and does not establish a medical "
                f"diagnosis."
            )


    # ========================================================
    # DIAGNOSIS QUESTION
    # ========================================================

    if "diagnosis" in message or "diagnose" in message:

        if prediction:

            return (
                f"The AI system predicted {class_name}, but "
                f"this does not constitute a confirmed diagnosis. "
                f"The model is designed for preliminary research "
                f"image analysis. A qualified healthcare "
                f"professional should evaluate the skin finding."
            )

        return (
            "I can explain the AI image-analysis system, but I "
            "cannot provide or confirm a medical diagnosis. "
            "A qualified healthcare professional should evaluate "
            "concerning skin findings."
        )


    # ========================================================
    # GENERAL DERMATOLOGY QUESTION
    # ========================================================

    if any(word in message for word in [
        "skin",
        "lesion",
        "mole",
        "rash",
        "spot",
        "itch",
        "bleeding",
    ]):

        return (
            "I can provide general information about dermatology "
            "and explain the AI image-analysis result. However, "
            "symptoms and skin findings can have many possible "
            "causes, so a qualified healthcare professional should "
            "evaluate persistent or concerning findings."
        )


    # ========================================================
    # DEFAULT
    # ========================================================

    if prediction:

        return (
            f"The current AI analysis predicts {class_name} "
            f"({prediction}) with "
            f"{format_confidence(confidence)} confidence. "
            f"I can explain what this prediction means, explain "
            f"the confidence, discuss the predicted class, or "
            f"explain what the next steps could be."
        )


    return (
        "I can help explain the AI image-analysis result, "
        "supported dermatology classes, confidence, uncertainty "
        "screening, and general dermatology information. "
        "Please ask a specific question."
    )


# ============================================================
# API ENDPOINT
# ============================================================

@router.post("/chat")
async def chat(request: ChatRequest):

    response = generate_response(request)

    return {
        "response": response,
        "agent": "Dermatology AI Assistant",
    }