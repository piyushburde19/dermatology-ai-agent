import pandas as pd
from pathlib import Path


VALIDATION_FILE = Path(
    "data/ood/validation_ood_results.csv"
)

EXTERNAL_FILE = Path(
    "data/ood/external_ood_results.csv"
)

OUTPUT_FILE = Path(
    "data/ood/ood_category_check.csv"
)


# Final candidate rule from the sweep
SIMILARITY_THRESHOLD = 0.37
CONFIDENCE_THRESHOLD = 0.85
ENTROPY_THRESHOLD = 0.60


print("=" * 70)
print("FINAL OOD CATEGORY CHECK")
print("=" * 70)

validation = pd.read_csv(
    VALIDATION_FILE
)

external = pd.read_csv(
    EXTERNAL_FILE
)


def apply_final_rule(df):

    low_similarity = (
        df["similarity"] < SIMILARITY_THRESHOLD
    )

    low_confidence = (
        df["confidence"] < CONFIDENCE_THRESHOLD
    )

    high_entropy = (
        df["entropy"] > ENTROPY_THRESHOLD
    )

    return (
        low_similarity
        & low_confidence
        & high_entropy
    )


validation["final_ood"] = apply_final_rule(
    validation
)

external["final_ood"] = apply_final_rule(
    external
)


# ------------------------------------------------------------
# VALIDATION BY DIAGNOSIS
# ------------------------------------------------------------

print()
print("=" * 70)
print("PAD-UFES VALIDATION BY DIAGNOSIS")
print("=" * 70)

validation_summary = (
    validation
    .groupby("diagnostic")
    .agg(
        images=("final_ood", "size"),
        ood_detected=("final_ood", "sum"),
    )
    .reset_index()
)

validation_summary["ood_rate_%"] = (
    validation_summary["ood_detected"]
    / validation_summary["images"]
    * 100
)

validation_summary = validation_summary.sort_values(
    "ood_rate_%",
    ascending=False,
)

print(
    validation_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# EXTERNAL BY CATEGORY
# ------------------------------------------------------------

print()
print("=" * 70)
print("SCIN EXTERNAL BY CATEGORY")
print("=" * 70)

external_summary = (
    external
    .groupby("category")
    .agg(
        images=("final_ood", "size"),
        ood_detected=("final_ood", "sum"),
    )
    .reset_index()
)

external_summary["ood_rate_%"] = (
    external_summary["ood_detected"]
    / external_summary["images"]
    * 100
)

external_summary = external_summary.sort_values(
    "ood_rate_%",
    ascending=False,
)

print(
    external_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# OVERALL
# ------------------------------------------------------------

print()
print("=" * 70)
print("OVERALL RESULT")
print("=" * 70)

print(
    f"PAD-UFES images: {len(validation)}"
)

print(
    f"PAD-UFES OOD: "
    f"{validation['final_ood'].sum()}"
)

print(
    f"PAD-UFES rejection: "
    f"{validation['final_ood'].mean() * 100:.2f}%"
)

print()

print(
    f"SCIN images: {len(external)}"
)

print(
    f"SCIN OOD: "
    f"{external['final_ood'].sum()}"
)

print(
    f"SCIN rejection: "
    f"{external['final_ood'].mean() * 100:.2f}%"
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

validation["dataset"] = "PAD-UFES-20"
external["dataset"] = "SCIN"

combined = pd.concat(
    [
        validation,
        external,
    ],
    ignore_index=True,
)

combined.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print(
    f"CSV: {OUTPUT_FILE}"
)

print()
print("FINAL RULE:")
print(
    f"Similarity < {SIMILARITY_THRESHOLD}"
)

print("AND")

print(
    f"Confidence < {CONFIDENCE_THRESHOLD}"
)

print("AND")

print(
    f"Entropy > {ENTROPY_THRESHOLD}"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)