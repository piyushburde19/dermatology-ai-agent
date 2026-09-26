import pandas as pd
from pathlib import Path


VALIDATION_FILE = Path(
    "data/ood/validation_ood_results.csv"
)

EXTERNAL_FILE = Path(
    "data/ood/external_ood_results.csv"
)


print("=" * 70)
print("COMBINED OOD SIGNAL ANALYSIS")
print("=" * 70)


validation = pd.read_csv(
    VALIDATION_FILE
)

external = pd.read_csv(
    EXTERNAL_FILE
)


print(
    f"Validation images: {len(validation)}"
)

print(
    f"External images:   {len(external)}"
)


# ------------------------------------------------------------
# BASIC COMPARISON
# ------------------------------------------------------------

print()
print("=" * 70)
print("SIGNAL COMPARISON")
print("=" * 70)


for column in [
    "similarity",
    "confidence",
    "entropy",
]:

    print()
    print(column.upper())

    print(
        f"PAD-UFES mean:   "
        f"{validation[column].mean():.4f}"
    )

    print(
        f"PAD-UFES median: "
        f"{validation[column].median():.4f}"
    )

    print(
        f"SCIN mean:       "
        f"{external[column].mean():.4f}"
    )

    print(
        f"SCIN median:     "
        f"{external[column].median():.4f}"
    )


# ------------------------------------------------------------
# CANDIDATE THRESHOLDS
# ------------------------------------------------------------

similarity_thresholds = [
    0.30,
    0.31,
    0.32,
    0.33,
    0.34,
    0.35,
    0.36,
    0.37,
    0.38,
    0.39,
    0.40,
]


confidence_thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
]


entropy_thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]


# ------------------------------------------------------------
# HELPER
# ------------------------------------------------------------

def evaluate_rule(
    df,
    similarity_threshold,
    confidence_threshold,
    entropy_threshold,
    mode,
):

    low_similarity = (
        df["similarity"]
        < similarity_threshold
    )

    low_confidence = (
        df["confidence"]
        < confidence_threshold
    )

    high_entropy = (
        df["entropy"]
        > entropy_threshold
    )


    if mode == "OR":

        detected = (
            low_similarity
            | low_confidence
            | high_entropy
        )

    elif mode == "AND":

        detected = (
            low_similarity
            & low_confidence
            & high_entropy
        )

    elif mode == "2_OF_3":

        detected = (
            low_similarity.astype(int)
            + low_confidence.astype(int)
            + high_entropy.astype(int)
            >= 2
        )

    else:

        raise ValueError(
            "Unknown mode"
        )


    return detected.mean() * 100


# ------------------------------------------------------------
# SINGLE SIGNAL BASELINES
# ------------------------------------------------------------

print()
print("=" * 70)
print("SINGLE SIGNAL BASELINES")
print("=" * 70)


print()
print("Similarity threshold = 0.3139")

print(
    "PAD-UFES rejection: "
    f"{evaluate_rule(validation, 0.3139, 0, 999, 'OR'):.2f}%"
)

print(
    "SCIN rejection:     "
    f"{evaluate_rule(external, 0.3139, 0, 999, 'OR'):.2f}%"
)


# ------------------------------------------------------------
# COMBINED RULE SWEEP
# ------------------------------------------------------------

results = []


for similarity_threshold in similarity_thresholds:

    for confidence_threshold in confidence_thresholds:

        for entropy_threshold in entropy_thresholds:

            for mode in [
                "OR",
                "AND",
                "2_OF_3",
            ]:

                validation_rejection = evaluate_rule(
                    validation,
                    similarity_threshold,
                    confidence_threshold,
                    entropy_threshold,
                    mode,
                )


                external_rejection = evaluate_rule(
                    external,
                    similarity_threshold,
                    confidence_threshold,
                    entropy_threshold,
                    mode,
                )


                results.append({

                    "mode": mode,

                    "similarity_threshold":
                        similarity_threshold,

                    "confidence_threshold":
                        confidence_threshold,

                    "entropy_threshold":
                        entropy_threshold,

                    "validation_rejection_%":
                        round(
                            validation_rejection,
                            2,
                        ),

                    "external_rejection_%":
                        round(
                            external_rejection,
                            2,
                        ),

                    "rejection_gap_%":
                        round(
                            external_rejection
                            - validation_rejection,
                            2,
                        ),

                })


results_df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# SAVE ALL RESULTS
# ------------------------------------------------------------

output_file = Path(
    "data/ood/combined_ood_sweep.csv"
)


results_df.to_csv(
    output_file,
    index=False,
)


print()
print("=" * 70)
print("SWEEP COMPLETE")
print("=" * 70)


print(
    f"Total rules tested: "
    f"{len(results_df)}"
)


print(
    f"Saved to: {output_file}"
)


# ------------------------------------------------------------
# SHOW RULES WITH LOW VALIDATION REJECTION
# ------------------------------------------------------------

print()
print("=" * 70)
print("RULES WITH <= 10% VALIDATION REJECTION")
print("=" * 70)


acceptable = results_df[
    results_df[
        "validation_rejection_%"
    ] <= 10
].copy()


acceptable = acceptable.sort_values(
    by=[
        "external_rejection_%",
        "validation_rejection_%",
    ],
    ascending=[
        False,
        True,
    ],
)


if len(acceptable) == 0:

    print(
        "No rule found."
    )

else:

    print(
        acceptable.head(20).to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# BEST RULES BY MODE
# ------------------------------------------------------------

print()
print("=" * 70)
print("BEST RULES BY MODE")
print("=" * 70)


for mode in [
    "OR",
    "AND",
    "2_OF_3",
]:

    mode_df = results_df[
        results_df["mode"] == mode
    ]


    mode_acceptable = mode_df[
        mode_df[
            "validation_rejection_%"
        ] <= 10
    ]


    if len(mode_acceptable) == 0:

        print()
        print(
            f"{mode}: No rule <= 10% validation rejection"
        )

        continue


    best = mode_acceptable.sort_values(
        by=[
            "external_rejection_%",
            "validation_rejection_%",
        ],
        ascending=[
            False,
            True,
        ],
    ).iloc[0]


    print()

    print(
        f"MODE: {mode}"
    )

    print(
        f"Similarity threshold: "
        f"{best['similarity_threshold']:.2f}"
    )

    print(
        f"Confidence threshold: "
        f"{best['confidence_threshold']:.2f}"
    )

    print(
        f"Entropy threshold: "
        f"{best['entropy_threshold']:.2f}"
    )

    print(
        f"PAD-UFES rejection: "
        f"{best['validation_rejection_%']:.2f}%"
    )

    print(
        f"SCIN rejection: "
        f"{best['external_rejection_%']:.2f}%"
    )

    print(
        f"Gap: "
        f"{best['rejection_gap_%']:.2f}%"
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)