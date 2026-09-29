from typing import Dict, Any

from backend.app.agents.image_analysis_agent import (
    image_analysis_agent,
)

from backend.app.agents.risk_uncertainty_agent import (
    risk_uncertainty_agent,
)

from backend.app.agents.explanation_agent import (
    explanation_agent,
)

from backend.app.agents.report_agent import (
    report_agent,
)


class DermatologyAgentOrchestrator:
    """
    Agent Orchestrator

    Coordinates the complete dermatology analysis pipeline.

    Pipeline:

    1. Image Analysis Agent
    2. Risk & Uncertainty Agent
    3. Explanation Agent
    4. Report Agent
    """

    name = "Dermatology Agent Orchestrator"

    def analyze(self, image_path: str) -> Dict[str, Any]:

        # ---------------------------------------------
        # 1. Image Analysis Agent
        # ---------------------------------------------

        image_analysis = image_analysis_agent.analyze(
            image_path
        )

        # ---------------------------------------------
        # 2. Risk & Uncertainty Agent
        # ---------------------------------------------

        risk_analysis = risk_uncertainty_agent.assess(
            image_analysis
        )

        # ---------------------------------------------
        # 3. Explanation Agent
        # ---------------------------------------------

        explanation = explanation_agent.explain(
            image_analysis,
            risk_analysis,
        )

        # ---------------------------------------------
        # 4. Report Agent
        # ---------------------------------------------

        report = report_agent.generate_report(
            image_analysis,
            risk_analysis,
            explanation,
        )

        # ---------------------------------------------
        # 5. Combine all agent outputs
        # ---------------------------------------------

        return {
            "orchestrator": self.name,

            "image_analysis": image_analysis,

            "risk_uncertainty": risk_analysis,

            "explanation": explanation,

            "report": report,

            "status": "completed",
        }


# --------------------------------------------------
# Shared orchestrator instance
# --------------------------------------------------

dermatology_orchestrator = DermatologyAgentOrchestrator()