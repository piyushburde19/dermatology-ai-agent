import pandas as pd
import requests
from pathlib import Path

CSV_PATH = Path("data/ood/scin_cases.csv")
OUTPUT_DIR = Path("data/ood/external/scin")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

categories = [
    "ACNE",
    "RASH",
    "LOOKS_HEALTHY",
    "OTHER_ISSUE_DESCRIPTION",
]

df = pd.read_csv(
    CSV_PATH,
    usecols=[
        "case_id",
        "related_category",
        "image_1_path",
        "image_2_path",
        "image_3_path",
    ],
)

selected = []

for category in categories:
    subset = df[df["related_category"] == category].head(10)
    selected.append(subset)

selected = pd.concat(selected, ignore_index=True)

base_url = "https://storage.googleapis.com/dx-scin-public-data/"

downloaded = 0
failed = 0

for _, row in selected.iterrows():

    category = row["related_category"]

    for image_col in ["image_1_path", "image_2_path", "image_3_path"]:

        image_path = row[image_col]

        if pd.isna(image_path) or not str(image_path).strip():
            continue

        filename = Path(str(image_path)).name
        output_path = OUTPUT_DIR / f"{category}_{row['case_id']}_{filename}"

        if output_path.exists():
            continue

        url = base_url + str(image_path)

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            output_path.write_bytes(response.content)

            downloaded += 1
            print(f"Downloaded: {output_path.name}")

        except Exception as e:
            failed += 1
            print(f"FAILED: {url}")
            print(f"Reason: {e}")

print()
print("=" * 50)
print("SCIN OOD DOWNLOAD COMPLETE")
print("=" * 50)
print(f"Downloaded: {downloaded}")
print(f"Failed:     {failed}")
print(f"Folder:     {OUTPUT_DIR}")