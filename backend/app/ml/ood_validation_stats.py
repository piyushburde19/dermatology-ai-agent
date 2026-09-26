import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from backend.app.ml.model import DermatologyEfficientNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path(
    "models/checkpoints/efficientnet_b0_moderated_best.pth"
)

OOD_CONFIG_PATH = Path(
    "models/checkpoints/ood_config.json"
)

METADATA_PATH = Path(
    "data/raw/PAD-UFES-20/metadata_split.csv"
)

IMAGE_FOLDER = Path(
    "data/raw/PAD-UFES-20/images"
)

RESULTS_FILE = Path(
    "data/ood/validation_ood_results.csv"
)


CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


print("=" * 60)
print("LOADING MODEL")
print("=" * 60)


model = DermatologyEfficientNet(
    num_classes=6,
    freeze_backbone=False,
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")


with open(
    OOD_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:
    config = json.load(file)


threshold = config["threshold"]


centroids = torch.tensor(
    config["centroids"],
    dtype=torch.float32,
    device=DEVICE,
)

centroids = F.normalize(
    centroids,
    p=2,
    dim=1,
)


print()
print("=" * 60)
print("LOADING VALIDATION DATA")
print("=" * 60)


metadata = pd.read_csv(
    METADATA_PATH
)


val_df = metadata[
    metadata["split"].str.lower() == "val"
].copy()


print(f"Validation images in metadata: {len(val_df)}")
print(f"OOD threshold: {threshold:.4f}")


results = []

missing_images = 0


print()
print("=" * 60)
print("EVALUATING VALIDATION IMAGES")
print("=" * 60)


for index, row in val_df.iterrows():

    img_id = str(row["img_id"])

    matches = list(
        IMAGE_FOLDER.rglob(img_id)
    )

    if not matches:

        print(
            f"MISSING: {img_id}"
        )

        missing_images += 1
        continue


    image_path = matches[0]


    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as error:

        print(
            f"FAILED: {img_id}"
        )

        print(
            f"Reason: {error}"
        )

        continue


    tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)


    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )


        confidence, predicted_index = torch.max(
            probabilities,
            dim=1,
        )


        entropy = -torch.sum(
            probabilities
            * torch.log(
                probabilities + 1e-8
            )
        ).item()


        features = model.model.features(
            tensor
        )

        features = model.model.avgpool(
            features
        )

        features = torch.flatten(
            features,
            1,
        )

        features = F.normalize(
            features,
            p=2,
            dim=1,
        )


        similarities = torch.mm(
            features,
            centroids.T,
        )


        similarity = similarities.max().item()


    predicted_class = CLASS_NAMES[
        predicted_index.item()
    ]


    is_ood = (
        similarity < threshold
    )


    results.append({

        "filename": img_id,

        "diagnostic": row["diagnostic"],

        "predicted_class": predicted_class,

        "similarity": round(
            similarity,
            4,
        ),

        "confidence": round(
            confidence.item(),
            4,
        ),

        "entropy": round(
            entropy,
            4,
        ),

        "threshold": round(
            threshold,
            4,
        ),

        "is_ood": is_ood,

    })


results_df = pd.DataFrame(
    results
)


RESULTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


results_df.to_csv(
    RESULTS_FILE,
    index=False,
)


print()
print("=" * 60)
print("VALIDATION OOD ANALYSIS COMPLETE")
print("=" * 60)


print(
    f"Validation images found: {len(results_df)}"
)

print(
    f"Missing images: {missing_images}"
)


if len(results_df) > 0:

    print()

    print("SIMILARITY")
    print(
        f"Mean:   {results_df['similarity'].mean():.4f}"
    )
    print(
        f"Median: {results_df['similarity'].median():.4f}"
    )
    print(
        f"Min:    {results_df['similarity'].min():.4f}"
    )
    print(
        f"Max:    {results_df['similarity'].max():.4f}"
    )

    print()

    print("CONFIDENCE")
    print(
        f"Mean:   {results_df['confidence'].mean():.4f}"
    )
    print(
        f"Median: {results_df['confidence'].median():.4f}"
    )
    print(
        f"Min:    {results_df['confidence'].min():.4f}"
    )
    print(
        f"Max:    {results_df['confidence'].max():.4f}"
    )

    print()

    print("ENTROPY")
    print(
        f"Mean:   {results_df['entropy'].mean():.4f}"
    )
    print(
        f"Median: {results_df['entropy'].median():.4f}"
    )
    print(
        f"Min:    {results_df['entropy'].min():.4f}"
    )
    print(
        f"Max:    {results_df['entropy'].max():.4f}"
    )

    print()

    print(
        "Current OOD rejection: "
        f"{results_df['is_ood'].mean() * 100:.2f}%"
    )


print()
print(
    f"CSV saved to: {RESULTS_FILE}"
)