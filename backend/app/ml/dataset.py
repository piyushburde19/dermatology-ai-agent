from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PADUFES20Dataset(Dataset):
    CLASS_NAMES = [
        "ACK",
        "BCC",
        "MEL",
        "NEV",
        "SCC",
        "SEK",
    ]

    CLASS_TO_IDX = {
        name: idx
        for idx, name in enumerate(CLASS_NAMES)
    }

    def __init__(
        self,
        metadata_path,
        image_root,
        split="train",
        image_size=224,
        augment=False,
    ):
        self.metadata_path = Path(metadata_path)
        self.image_root = Path(image_root)
        self.split = split

        self.df = pd.read_csv(self.metadata_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.image_size = image_size
        self.augment = augment

        self.transform = self._build_transforms()

    def _build_transforms(self):
        if self.augment and self.split == "train":
            return transforms.Compose([
                transforms.Resize(
                    (self.image_size, self.image_size)
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5
                ),
                transforms.RandomVerticalFlip(
                    p=0.2
                ),
                transforms.RandomRotation(
                    degrees=15
                ),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        return transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _find_image(self, img_id):
        matches = list(
            self.image_root.rglob(img_id)
        )

        if not matches:
            raise FileNotFoundError(
                f"Image not found: {img_id}"
            )

        return matches[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = self._find_image(
            row["img_id"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        label = self.CLASS_TO_IDX[
            row["diagnostic"]
        ]

        return {
            "image": image,
            "label": label,
            "img_id": row["img_id"],
            "patient_id": row["patient_id"],
            "lesion_id": row["lesion_id"],
            "diagnostic": row["diagnostic"],
        }