import torch
import pandas as pd

from torch.utils.data import DataLoader

from backend.app.ml.dataset import PADUFES20Dataset


CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
]


def create_dataloaders(
    metadata_path="data/raw/PAD-UFES-20/metadata_split.csv",
    image_root="data/raw/PAD-UFES-20/images",
    batch_size=8,
    num_workers=0,
):
    # -------------------------
    # Create datasets
    # -------------------------

    train_dataset = PADUFES20Dataset(
        metadata_path=metadata_path,
        image_root=image_root,
        split="train",
        image_size=224,
        augment=True,
    )

    val_dataset = PADUFES20Dataset(
        metadata_path=metadata_path,
        image_root=image_root,
        split="val",
        image_size=224,
        augment=False,
    )

    test_dataset = PADUFES20Dataset(
        metadata_path=metadata_path,
        image_root=image_root,
        split="test",
        image_size=224,
        augment=False,
    )

    # -------------------------
    # Calculate class weights
    # -------------------------

    labels = train_dataset.df["diagnostic"].map(
        train_dataset.CLASS_TO_IDX
    )

    class_counts = torch.bincount(
        torch.tensor(labels.values),
        minlength=len(CLASS_NAMES),
    ).float()

    class_weights = torch.sqrt(
    class_counts.sum() / (
        len(CLASS_NAMES) * class_counts
    )
)

    # -------------------------
    # DataLoaders
    # -------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_weights,
    )