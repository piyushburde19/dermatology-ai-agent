import json
from pathlib import Path

import pandas as pd
import requests


API_URL = "http://127.0.0.1:8000/api/v1/upload"

DATASET_DIR = Path(
    "data/raw/PAD-UFES-20"
)

METADATA_PATH = (
    DATASET_DIR / "metadata_split.csv"
)

IMAGE_DIR = (
    DATASET_DIR / "images"
)


CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
]


def find_image(img_id):
    matches = list(
        IMAGE_DIR.rglob(f"{img_id}*")
    )

    if not matches:
        return None

    return matches[0]


def main():

    df = pd.read_csv(
        METADATA_PATH
    )

    # Use only test split
    test_df = df[
        df["split"].str.lower() == "test"
    ]

    results = []

    print()
    print("=" * 70)
    print("MULTI-CLASS AGENT PIPELINE TEST")
    print("=" * 70)

    for class_name in CLASS_NAMES:

        samples = test_df[
            test_df["diagnostic"] == class_name
        ]

        if samples.empty:
            print(
                f"\n{class_name}: No test image found"
            )
            continue

        row = samples.iloc[0]

        img_id = str(
            row["img_id"]
        )

        image_path = find_image(
            img_id
        )

        if image_path is None:
            print(
                f"\n{class_name}: Image not found for {img_id}"
            )
            continue

        print()
        print("-" * 70)
        print(
            f"Actual Class : {class_name}"
        )
        print(
            f"Image        : {image_path.name}"
        )

        try:

            with open(
                image_path,
                "rb"
            ) as image_file:

                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            image_path.name,
                            image_file,
                            "image/png",
                        )
                    },
                    timeout=120,
                )

            response.raise_for_status()

            data = response.json()

            image_analysis = data.get(
                "image_analysis",
                {}
            )

            risk = data.get(
                "risk_uncertainty",
                {}
            )

            report = data.get(
                "report",
                {}
            )

            prediction = image_analysis.get(
                "prediction"
            )

            confidence = image_analysis.get(
                "confidence"
            )

            ood = image_analysis.get(
                "ood"
            )

            assessment = risk.get(
                "assessment"
            )

            print(
                f"Prediction   : {prediction}"
            )

            print(
                f"Confidence   : {confidence}%"
            )

            print(
                f"OOD          : {ood}"
            )

            print(
                f"Assessment   : {assessment}"
            )

            print(
                f"Report       : {report.get('status')}"
            )

            results.append({
                "actual_class": class_name,
                "image": image_path.name,
                "prediction": prediction,
                "confidence": confidence,
                "ood": ood,
                "assessment": assessment,
                "report_status": report.get(
                    "status"
                ),
            })

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

            results.append({
                "actual_class": class_name,
                "image": image_path.name,
                "prediction": "ERROR",
                "confidence": None,
                "ood": None,
                "assessment": "ERROR",
                "report_status": "ERROR",
            })

    output_path = Path(
        "tests/multi_class_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(
        f"Results saved to: {output_path}"
    )


if __name__ == "__main__":
    main()