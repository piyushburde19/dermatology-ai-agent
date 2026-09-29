import pandas as pd

from backend.app.agents.risk_uncertainty_agent import (
    risk_uncertainty_agent,
)


df = pd.read_csv("tests/test_set_ood_results.csv")


# Only images that are NOT OOD
supported = df[df["ood"] == False].copy()


print()
print("=" * 80)
print("REVIEW ROUTING VALIDATION")
print("=" * 80)

print(f"Total test images       : {len(df)}")
print(f"Supported images        : {len(supported)}")
print()


review_cases = []

for _, row in supported.iterrows():

    analysis_result = {
        "prediction": row["prediction"],
        "confidence": row["confidence"],
        "similarity": row["similarity"],
        "entropy": row["entropy"],
        "ood": False,
        "ood_thresholds": {
            "similarity": 0.37,
            "confidence": 0.85,
            "entropy": 0.60,
        },
    }

    result = risk_uncertainty_agent.assess(
        analysis_result
    )

    if result["review_required"]:

        review_cases.append({
            "actual_class": row["actual_class"],
            "image": row["image"],
            "prediction": row["prediction"],
            "confidence": row["confidence"],
            "similarity": row["similarity"],
            "entropy": row["entropy"],
            "assessment": result["assessment"],
            "signals": result["signals"],
        })


print("=" * 80)
print("REVIEW ROUTING RESULTS")
print("=" * 80)

print(f"Supported images              : {len(supported)}")
print(f"Review Recommended            : {len(review_cases)}")
print(
    f"Supported without review      : "
    f"{len(supported) - len(review_cases)}"
)

print()


if review_cases:

    review_df = pd.DataFrame(review_cases)

    print("=" * 80)
    print("REVIEW-RECOMMENDED CASES")
    print("=" * 80)

    print(
        review_df[
            [
                "actual_class",
                "image",
                "prediction",
                "confidence",
                "similarity",
                "entropy",
                "assessment",
            ]
        ].to_string(index=False)
    )

    print()

    print("=" * 80)
    print("REVIEW SIGNAL COUNTS")
    print("=" * 80)

    print(
        "Low confidence :",
        sum(
            x["signals"]["low_confidence"]
            for x in review_cases
        ),
    )

    print(
        "Low similarity :",
        sum(
            x["signals"]["low_similarity"]
            for x in review_cases
        ),
    )

    print(
        "High entropy   :",
        sum(
            x["signals"]["high_entropy"]
            for x in review_cases
        ),
    )

else:

    print("No Review Recommended cases found.")


print()
print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)