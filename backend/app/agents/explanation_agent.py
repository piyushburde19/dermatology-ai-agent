from typing import Dict, Any


class ExplanationAgent:
    """
    Explanation Agent

    Converts the outputs of the Image Analysis Agent and
    Risk & Uncertainty Agent into a human-readable explanation.

    This agent does not provide a definitive medical diagnosis.
    """

    name = "Explanation Agent"

    # ---------------------------------------------
    # Condition information
    # ---------------------------------------------

    CONDITION_INFO = {
        "ACK": {
            "name": "Actinic Keratosis",
            "description": (
                "a skin lesion commonly associated with "
                "long-term sun exposure."
            ),
        },

        "BCC": {
            "name": "Basal Cell Carcinoma",
            "description": (
                "a type of skin cancer arising from "
                "basal cells of the skin."
            ),
        },

        "MEL": {
            "name": "Melanoma",
            "description": (
                "a type of skin cancer involving "
                "melanocytes, the pigment-producing cells "
                "of the skin."
            ),
        },

        "NEV": {
            "name": "Nevus",
            "description": (
                "a commonly occurring pigmented skin lesion, "
                "often referred to as a mole."
            ),
        },

        "SCC": {
            "name": "Squamous Cell Carcinoma",
            "description": (
                "a type of skin cancer arising from "
                "squamous cells."
            ),
        },

        "SEK": {
            "name": "Seborrheic Keratosis",
            "description": (
                "a common benign skin growth that can "
                "appear as a raised or scaly lesion."
            ),
        },
    }

    def explain(
        self,
        image_analysis: Dict[str, Any],
        risk_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:

        # ---------------------------------------------
        # 1. Extract prediction
        # ---------------------------------------------

        prediction = image_analysis.get(
            "prediction"
        )

        confidence = image_analysis.get(
            "confidence", 0
        )

        similarity = image_analysis.get(
            "similarity", 0
        )

        entropy = image_analysis.get(
            "entropy", 0
        )

        ood = image_analysis.get(
            "ood", True
        )

        gradcam_available = image_analysis.get(
            "gradcam_available",
            False,
        )

        assessment = risk_analysis.get(
            "assessment",
            "Unknown",
        )

        review_required = risk_analysis.get(
            "review_required",
            True,
        )

        # ---------------------------------------------
        # 2. Handle unsupported image
        # ---------------------------------------------

        if ood:

            summary = (
                "The uploaded image could not be "
                "reliably matched to the supported "
                "dermatology image distribution."
            )

            interpretation = (
                "The model's prediction should not be "
                "interpreted as a reliable classification "
                "for this image."
            )

            next_step = (
                "Use a suitable dermatology lesion image "
                "within the supported image domain or seek "
                "evaluation from a qualified healthcare "
                "professional."
            )

            return {
                "agent": self.name,
                "summary": summary,
                "interpretation": interpretation,
                "next_step": next_step,
                "prediction": prediction,
                "confidence": confidence,
                "assessment": assessment,
                "review_required": review_required,
                "gradcam_available": False,
                "status": "completed",
            }

        # ---------------------------------------------
        # 3. Get condition information
        # ---------------------------------------------

        condition = self.CONDITION_INFO.get(
            prediction,
            {
                "name": prediction or "Unknown",
                "description": (
                    "a dermatology classification "
                    "returned by the model."
                ),
            },
        )

        condition_name = condition["name"]
        condition_description = condition[
            "description"
        ]

        # ---------------------------------------------
        # 4. Build summary
        # ---------------------------------------------

        summary = (
            f"The image analysis model classified the "
            f"lesion as {condition_name} ({prediction}) "
            f"with a confidence of {confidence:.2f}%."
        )

        # ---------------------------------------------
        # 5. Build interpretation
        # ---------------------------------------------

        interpretation = (
            f"{condition_name} is {condition_description} "
            f"The model's feature similarity was "
            f"{similarity:.4f}, while prediction entropy "
            f"was {entropy:.4f}. "
            f"The uncertainty assessment was "
            f"'{assessment}'."
        )

        # ---------------------------------------------
        # 6. Review recommendation
        # ---------------------------------------------

        if review_required:

            next_step = (
                "Because uncertainty signals were detected, "
                "the result should be reviewed carefully "
                "and should not be treated as a definitive "
                "diagnosis."
            )

        else:

            next_step = (
                "The model did not detect major uncertainty "
                "signals for this image. However, this is "
                "an AI-assisted result and not a definitive "
                "medical diagnosis."
            )

        # ---------------------------------------------
        # 7. Grad-CAM explanation
        # ---------------------------------------------

        if gradcam_available:

            explanation_note = (
                "A Grad-CAM visualization is available. "
                "It highlights image regions that contributed "
                "to the model's prediction. These highlighted "
                "regions represent model attention and do "
                "not prove clinical diagnosis."
            )

        else:

            explanation_note = (
                "A Grad-CAM visualization was not available "
                "for this analysis."
            )

        # ---------------------------------------------
        # 8. Return structured explanation
        # ---------------------------------------------

        return {
            "agent": self.name,

            "prediction": prediction,

            "condition_name": condition_name,

            "confidence": confidence,

            "assessment": assessment,

            "review_required": review_required,

            "summary": summary,

            "interpretation": interpretation,

            "next_step": next_step,

            "gradcam_available": gradcam_available,

            "explanation_note": explanation_note,

            "status": "completed",
        }


# --------------------------------------------------
# Shared agent instance
# --------------------------------------------------

explanation_agent = ExplanationAgent()