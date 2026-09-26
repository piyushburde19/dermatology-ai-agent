import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from backend.app.ml.model import DermatologyEfficientNet


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path(
    "models/checkpoints/efficientnet_b0_moderated_best.pth"
)

OOD_CONFIG_PATH = Path(
    "models/checkpoints/ood_config.json"
)

OOD_FOLDER = Path(
    "data/ood/external"
)

RESULTS_FILE = Path(
    "data/ood/external_ood_results.csv"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# LOAD MODEL
# ============================================================

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


# ============================================================
# LOAD OOD CONFIGURATION
# ============================================================

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


# ============================================================
# OOD EVALUATION
# ============================================================

print()
print("=" * 60)
print("OOD EVALUATION")
print("=" * 60)

print(f"Threshold: {threshold:.4f}")
print(f"OOD folder: {OOD_FOLDER}")
print()


results = []


# Recursively scan all image files
for image_path in sorted(
    OOD_FOLDER.rglob("*")
):

    if image_path.suffix.lower() not in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    ]:
        continue


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    try:
        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as error:
        print(
            f"FAILED TO OPEN: {image_path.name}"
        )
        print(
            f"Reason: {error}"
        )
        continue


    tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)


    # --------------------------------------------------------
    # MODEL PREDICTION + FEATURE EXTRACTION
    # --------------------------------------------------------

    with torch.no_grad():

        # Classification logits
        logits = model(tensor)

        # Class probabilities
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        # Maximum class probability
        confidence = probabilities.max().item()

        # Prediction entropy
        entropy = -torch.sum(
            probabilities
            * torch.log(
                probabilities + 1e-8
            )
        ).item()


        # EfficientNet feature extraction
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


        # Cosine similarity against all class centroids
        similarities = torch.mm(
            features,
            centroids.T,
        )

        # Maximum similarity to any supported class
        similarity = similarities.max().item()


    # --------------------------------------------------------
    # OOD DECISION
    # --------------------------------------------------------

    is_ood = similarity < threshold

    status = (
        "OOD"
        if is_ood
        else "SUPPORTED"
    )


    # --------------------------------------------------------
    # DETECT SOURCE CATEGORY
    # --------------------------------------------------------

    filename_upper = image_path.name.upper()

    if filename_upper.startswith(
        "ACNE_"
    ):
        category = "ACNE"

    elif filename_upper.startswith(
        "RASH_"
    ):
        category = "RASH"

    elif filename_upper.startswith(
        "LOOKS_HEALTHY_"
    ):
        category = "LOOKS_HEALTHY"

    elif filename_upper.startswith(
        "OTHER_ISSUE_DESCRIPTION_"
    ):
        category = "OTHER_ISSUE_DESCRIPTION"

    elif filename_upper == "ACNE.PNG":

        category = "ACNE"

    else:
        category = "UNKNOWN"


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append({
        "filename": image_path.name,
        "category": category,
        "similarity": round(
            similarity,
            4,
        ),
        "confidence": round(
            confidence,
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
        "status": status,
    })


    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print(
        f"{image_path.name:70} "
        f"sim={similarity:.4f} "
        f"conf={confidence:.4f} "
        f"entropy={entropy:.4f} "
        f"{status}"
    )


# ============================================================
# SAVE CSV
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("OOD EVALUATION COMPLETE")
print("=" * 60)

print(
    f"Images evaluated: {len(results_df)}"
)

if len(results_df) > 0:

    print(
        f"OOD detected: "
        f"{results_df['is_ood'].sum()}"
    )

    print(
        f"Supported: "
        f"{(~results_df['is_ood']).sum()}"
    )

    print()

    print(
        "Overall OOD rejection rate: "
        f"{results_df['is_ood'].mean() * 100:.2f}%"
    )

print()

print(
    f"CSV saved to: {RESULTS_FILE}"
)