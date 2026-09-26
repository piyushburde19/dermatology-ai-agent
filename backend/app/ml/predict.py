import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from backend.app.ml.model import DermatologyEfficientNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
]


CHECKPOINT_PATH = Path(
    "models/checkpoints/efficientnet_b0_moderated_best.pth"
)

OOD_CONFIG_PATH = Path(
    "models/checkpoints/ood_config.json"
)


# ------------------------------------------------------------
# IMAGE TRANSFORM
# ------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# LOAD OOD CONFIG
# ------------------------------------------------------------

with open(
    OOD_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:

    OOD_CONFIG = json.load(file)


CENTROIDS = torch.tensor(
    OOD_CONFIG["centroids"],
    dtype=torch.float32,
    device=DEVICE,
)

CENTROIDS = F.normalize(
    CENTROIDS,
    p=2,
    dim=1,
)


# ------------------------------------------------------------
# FINAL OOD RULE
# ------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.37
CONFIDENCE_THRESHOLD = 0.85
ENTROPY_THRESHOLD = 0.60


# ------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_skin_disease(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    tensor = TRANSFORM(
        image
    ).unsqueeze(0).to(DEVICE)


    with torch.no_grad():

        # --------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------

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


        # --------------------------------------------
        # ENTROPY
        # --------------------------------------------

        entropy = -torch.sum(
            probabilities
            * torch.log(
                probabilities + 1e-8
            )
        ).item()


        # --------------------------------------------
        # FEATURE EXTRACTION
        # --------------------------------------------

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


        # --------------------------------------------
        # COSINE SIMILARITY
        # --------------------------------------------

        similarities = torch.mm(
            features,
            CENTROIDS.T,
        )

        similarity, _ = torch.max(
            similarities,
            dim=1,
        )


    # ------------------------------------------------
    # CONVERT VALUES
    # ------------------------------------------------

    predicted_index = predicted_index.item()

    confidence = confidence.item()

    similarity = similarity.item()


    raw_prediction = CLASS_NAMES[
        predicted_index
    ]


    # ------------------------------------------------
    # FINAL OOD DECISION
    # ------------------------------------------------

    low_similarity = (
        similarity < SIMILARITY_THRESHOLD
    )

    low_confidence = (
        confidence < CONFIDENCE_THRESHOLD
    )

    high_entropy = (
        entropy > ENTROPY_THRESHOLD
    )


    is_ood = (
        low_similarity
        and low_confidence
        and high_entropy
    )


    if is_ood:

        final_prediction = (
            "Unsupported / Uncertain"
        )

    else:

        final_prediction = raw_prediction


    # ------------------------------------------------
    # PROBABILITIES
    # ------------------------------------------------

    probability_dict = {

        CLASS_NAMES[i]:
            round(
                probabilities[0][i].item() * 100,
                2,
            )

        for i in range(
            len(CLASS_NAMES)
        )

    }


    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------

    return {

        "prediction":
            final_prediction,

        "raw_prediction":
            raw_prediction,

        "confidence":
            round(
                confidence * 100,
                2,
            ),

        "probabilities":
            probability_dict,

        "ood":
            is_ood,

        "similarity":
            round(
                similarity,
                4,
            ),

        "entropy":
            round(
                entropy,
                4,
            ),

        "ood_thresholds": {

            "similarity":
                SIMILARITY_THRESHOLD,

            "confidence":
                CONFIDENCE_THRESHOLD,

            "entropy":
                ENTROPY_THRESHOLD,

        },

    }