import json
from pathlib import Path

import pandas as pd

from backend.app.ml.predict import predict_skin_disease


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATASET_DIR = Path(
    "data/raw/PAD-UFES-20"
)

METADATA_PATH = (
    DATASET_DIR / "metadata_split.csv"
)

IMAGE_DIR = (
    DATASET_DIR / "images"
)

OUTPUT_CSV = Path(
    "tests/test_set_ood_results.csv"
)

OUTPUT_JSON = Path(
    "tests/test_set_ood_summary.json"
)


CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
]


# --------------------------------------------------
# Find image
# --------------------------------------------------

def find_image(img_id):

    matches = list(
        IMAGE_DIR.rglob(f"{img_id}*")
    )

    if not matches:
        return None

    return matches[0]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    df = pd.read_csv(
        METADATA_PATH
    )

    # Only held-out test set
    test_df = df[
        df["split"].str.lower() == "test"
    ].copy()

    print()
    print("=" * 80)
    print("FULL TEST-SET OOD VALIDATION")
    print("=" * 80)

    print(
        f"Test images: {len(test_df)}"
    )

    results = []

    # --------------------------------------------------
    # Run prediction on every test image
    # --------------------------------------------------

    for index, row in test_df.iterrows():

        actual_class = str(
            row["diagnostic"]
        )

        img_id = str(
            row["img_id"]
        )

        image_path = find_image(
            img_id
        )

        if image_path is None:

            print(
                f"[SKIP] Image not found: {img_id}"
            )

            continue

        try:

            result = predict_skin_disease(
                str(image_path)
            )

            prediction = result.get(
                "prediction"
            )

            raw_prediction = result.get(
                "raw_prediction"
            )

            confidence = result.get(
                "confidence"
            )

            similarity = result.get(
                "similarity"
            )

            entropy = result.get(
                "entropy"
            )

            ood = result.get(
                "ood"
            )

            results.append({

                "actual_class":
                    actual_class,

                "image":
                    image_path.name,

                "prediction":
                    prediction,

                "raw_prediction":
                    raw_prediction,

                "confidence":
                    confidence,

                "similarity":
                    similarity,

                "entropy":
                    entropy,

                "ood":
                    ood,

            })

        except Exception as error:

            print(
                f"[ERROR] {image_path.name}: {error}"
            )

    # --------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # Save individual results
    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------
    # Overall statistics
    # --------------------------------------------------

    total = len(
        results_df
    )

    ood_count = int(
        results_df["ood"].sum()
    )

    supported_count = (
        total - ood_count
    )

    overall_ood_rate = (
        ood_count / total * 100
        if total > 0
        else 0
    )

    # Classification accuracy among
    # supported predictions only
    supported_df = results_df[
        results_df["ood"] == False
    ]

    supported_accuracy = (
        (
            supported_df["prediction"]
            == supported_df["actual_class"]
        ).mean() * 100
        if len(supported_df) > 0
        else 0
    )

    # --------------------------------------------------
    # Per-class OOD statistics
    # --------------------------------------------------

    class_summary = {}

    for class_name in CLASS_NAMES:

        class_df = results_df[
            results_df["actual_class"]
            == class_name
        ]

        class_total = len(
            class_df
        )

        class_ood = int(
            class_df["ood"].sum()
        )

        class_supported = (
            class_total - class_ood
        )

        class_ood_rate = (
            class_ood / class_total * 100
            if class_total > 0
            else 0
        )

        class_supported_df = class_df[
            class_df["ood"] == False
        ]

        class_accuracy = (
            (
                class_supported_df["prediction"]
                == class_supported_df["actual_class"]
            ).mean() * 100
            if len(class_supported_df) > 0
            else 0
        )

        class_summary[class_name] = {

            "total_images":
                class_total,

            "ood_images":
                class_ood,

            "supported_images":
                class_supported,

            "ood_rejection_rate":
                round(
                    class_ood_rate,
                    2
                ),

            "accuracy_on_supported":
                round(
                    class_accuracy,
                    2
                ),

        }

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = {

        "total_test_images":
            total,

        "supported_images":
            supported_count,

        "ood_images":
            ood_count,

        "overall_ood_rejection_rate":
            round(
                overall_ood_rate,
                2
            ),

        "accuracy_on_supported_images":
            round(
                supported_accuracy,
                2
            ),

        "similarity_threshold":
            0.37,

        "confidence_threshold":
            0.85,

        "entropy_threshold":
            0.60,

        "class_summary":
            class_summary,

    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    print(
        f"Total test images       : {total}"
    )

    print(
        f"Supported images        : {supported_count}"
    )

    print(
        f"OOD images              : {ood_count}"
    )

    print(
        f"OOD rejection rate      : "
        f"{overall_ood_rate:.2f}%"
    )

    print(
        f"Accuracy on supported   : "
        f"{supported_accuracy:.2f}%"
    )

    print()
    print("=" * 80)
    print("PER-CLASS OOD RESULTS")
    print("=" * 80)

    for class_name in CLASS_NAMES:

        info = class_summary[
            class_name
        ]

        print()

        print(
            f"{class_name}:"
        )

        print(
            f"  Total              : "
            f"{info['total_images']}"
        )

        print(
            f"  OOD                : "
            f"{info['ood_images']}"
        )

        print(
            f"  Supported          : "
            f"{info['supported_images']}"
        )

        print(
            f"  OOD rejection      : "
            f"{info['ood_rejection_rate']}%"
        )

        print(
            f"  Supported accuracy : "
            f"{info['accuracy_on_supported']}%"
        )

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    print(
        f"CSV: {OUTPUT_CSV}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()