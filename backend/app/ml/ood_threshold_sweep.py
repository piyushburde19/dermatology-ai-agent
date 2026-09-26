from pathlib import Path

import numpy as np
import pandas as pd


VALIDATION_FILE = Path(
    "data/ood/validation_ood_results.csv"
)

EXTERNAL_FILE = Path(
    "data/ood/external_ood_results.csv"
)


validation = pd.read_csv(VALIDATION_FILE)
external = pd.read_csv(EXTERNAL_FILE)


validation_scores = validation["similarity"].to_numpy()
external_scores = external["similarity"].to_numpy()


thresholds = [
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


rows = []


for threshold in thresholds:

    validation_rejection = (
        np.mean(validation_scores < threshold) * 100
    )

    external_rejection = (
        np.mean(external_scores < threshold) * 100
    )

    rows.append({
        "threshold": threshold,
        "validation_rejection_%": round(
            validation_rejection,
            2,
        ),
        "external_rejection_%": round(
            external_rejection,
            2,
        ),
    })


results = pd.DataFrame(rows)


print("=" * 70)
print("OOD THRESHOLD SWEEP")
print("=" * 70)

print()
print(
    results.to_string(index=False)
)


print()
print("=" * 70)
print("EXTERNAL REJECTION BY CATEGORY")
print("=" * 70)

category_rows = []

for threshold in thresholds:

    for category, group in external.groupby(
        "category"
    ):

        rejection = (
            np.mean(
                group["similarity"].to_numpy()
                < threshold
            )
            * 100
        )

        category_rows.append({
            "threshold": threshold,
            "category": category,
            "rejection_%": round(
                rejection,
                2,
            ),
        })


category_results = pd.DataFrame(
    category_rows
)

print()
print(
    category_results.to_string(
        index=False
    )
)


output_file = Path(
    "data/ood/ood_threshold_sweep.csv"
)

results.to_csv(
    output_file,
    index=False,
)

print()
print(
    f"Saved: {output_file}"
)
