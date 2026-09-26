import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.app.ml.dataset import PADUFES20Dataset
from backend.app.ml.model import DermatologyEfficientNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METADATA_PATH = "data/raw/PAD-UFES-20/metadata_split.csv"
IMAGE_ROOT = "data/raw/PAD-UFES-20/images"

CHECKPOINT_PATH = Path(
    "models/checkpoints/efficientnet_b0_moderated_best.pth"
)

OUTPUT_PATH = Path(
    "models/checkpoints/ood_config.json"
)

CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
]


def get_embeddings(model, loader):
    embeddings = []
    labels = []

    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)

            features = model.model.features(images)
            features = model.model.avgpool(features)
            features = torch.flatten(features, 1)

            features = F.normalize(features, p=2, dim=1)

            embeddings.append(features.cpu())
            labels.append(batch["label"])

    return torch.cat(embeddings), torch.cat(labels)


def main():

    print("=" * 60)
    print("BUILDING OOD CONFIGURATION")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print()
    print("Loading model...")

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

    print(
        f"Loaded checkpoint from epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print()
    print("Loading datasets...")

    train_dataset = PADUFES20Dataset(
        metadata_path=METADATA_PATH,
        image_root=IMAGE_ROOT,
        split="train",
        image_size=224,
        augment=False,
    )

    val_dataset = PADUFES20Dataset(
        metadata_path=METADATA_PATH,
        image_root=IMAGE_ROOT,
        split="val",
        image_size=224,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    print()
    print("Extracting training embeddings...")

    train_embeddings, train_labels = get_embeddings(
        model,
        train_loader,
    )

    print("Extracting validation embeddings...")

    val_embeddings, _ = get_embeddings(
        model,
        val_loader,
    )

    print()
    print("Building class centroids...")

    centroids = []

    for class_index in range(len(CLASS_NAMES)):

        class_embeddings = train_embeddings[
            train_labels == class_index
        ]

        centroid = class_embeddings.mean(dim=0)

        centroid = F.normalize(
            centroid.unsqueeze(0),
            p=2,
            dim=1,
        ).squeeze(0)

        centroids.append(centroid)

        print(
            f"{CLASS_NAMES[class_index]}: "
            f"{len(class_embeddings)} images"
        )

    centroids = torch.stack(centroids)

    print()
    print("Calculating validation similarity scores...")

    similarity_matrix = torch.mm(
        val_embeddings,
        centroids.T,
    )

    max_similarity = similarity_matrix.max(
        dim=1
    ).values

    threshold = torch.quantile(
        max_similarity,
        0.05,
    ).item()

    print()
    print("=" * 60)
    print("OOD CONFIGURATION")
    print("=" * 60)

    print(
        f"Minimum validation similarity: "
        f"{max_similarity.min().item():.4f}"
    )

    print(
        f"Maximum validation similarity: "
        f"{max_similarity.max().item():.4f}"
    )

    print(
        f"OOD threshold: "
        f"{threshold:.4f}"
    )

    config = {
        "method": "cosine_centroid",
        "threshold": threshold,
        "class_names": CLASS_NAMES,
        "checkpoint": str(CHECKPOINT_PATH),
        "centroids": centroids.tolist(),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
        )

    print()
    print(
        f"Saved OOD configuration to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()