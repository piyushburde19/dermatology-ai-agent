from typing import Dict, Any


class RiskUncertaintyAgent:
    """
    Risk & Uncertainty Agent

    This agent evaluates the reliability of the image-analysis
    model output using:

    - OOD status
    - prediction confidence
    - feature similarity
    - prediction entropy

    It does NOT provide a medical diagnosis or clinical risk score.
    """

    name = "Risk & Uncertainty Agent"

    def assess(self, analysis_result: Dict[str, Any]):
        # ---------------------------------------------
        # 1. Extract model outputs
        # ---------------------------------------------

        confidence = analysis_result.get(
            "confidence", 0
        )

        similarity = analysis_result.get(
            "similarity", 0
        )

        entropy = analysis_result.get(
            "entropy", 1
        )

        ood = analysis_result.get(
            "ood", True
        )

        thresholds = analysis_result.get(
            "ood_thresholds",
            {}
        )

        similarity_threshold = thresholds.get(
            "similarity", 0.37
        )

        confidence_threshold = thresholds.get(
            "confidence", 0.85
        )

        entropy_threshold = thresholds.get(
            "entropy", 0.60
        )

        # ---------------------------------------------
        # 2. Normalize confidence
        # ---------------------------------------------
        # Prediction pipeline returns confidence as
        # percentage (e.g. 96.13), while threshold
        # is stored as decimal (0.85).

        confidence_decimal = confidence

        if confidence > 1:
            confidence_decimal = confidence / 100.0

        # ---------------------------------------------
        # 3. Evaluate individual uncertainty signals
        # ---------------------------------------------

        low_confidence = (
            confidence_decimal < confidence_threshold
        )

        low_similarity = (
            similarity < similarity_threshold
        )

        high_entropy = (
            entropy > entropy_threshold
        )

        # ---------------------------------------------
        # 4. Determine overall assessment
        # ---------------------------------------------

        if ood:
            assessment = "Unsupported"
            review_required = True

            explanation = (
                "The uploaded image appears outside "
                "the supported image distribution of "
                "the dermatology model."
            )

        elif (
            low_confidence
            or low_similarity
            or high_entropy
        ):
            assessment = "Review Recommended"
            review_required = True

            signals = []

            if low_confidence:
                signals.append(
                    "low prediction confidence"
                )

            if low_similarity:
                signals.append(
                    "low feature similarity"
                )

            if high_entropy:
                signals.append(
                    "high prediction uncertainty"
                )

            explanation = (
                "The model produced a supported prediction, "
                "but uncertainty signals suggest that the "
                "result should be reviewed carefully. "
                f"Detected signals: {', '.join(signals)}."
            )

        else:
            assessment = "Supported"
            review_required = False

            explanation = (
                "The image is within the model's supported "
                "distribution and the prediction shows "
                "acceptable confidence and uncertainty "
                "signals."
            )

        # ---------------------------------------------
        # 5. Return structured agent result
        # ---------------------------------------------

        return {
            "agent": self.name,

            "assessment": assessment,

            "review_required": review_required,

            "confidence": confidence,

            "similarity": similarity,

            "entropy": entropy,

            "ood": ood,

            "signals": {
                "low_confidence": low_confidence,
                "low_similarity": low_similarity,
                "high_entropy": high_entropy,
            },

            "thresholds": {
                "confidence": confidence_threshold,
                "similarity": similarity_threshold,
                "entropy": entropy_threshold,
            },

            "explanation": explanation,

            "status": "completed",
        }


# --------------------------------------------------
# Shared agent instance
# --------------------------------------------------

risk_uncertainty_agent = RiskUncertaintyAgent()