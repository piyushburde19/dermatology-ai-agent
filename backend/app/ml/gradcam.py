from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from backend.app.ml.model import DermatologyEfficientNet


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = Path(
    "models/checkpoints/efficientnet_b0_moderated_best.pth"
)

METADATA_PATH = Path(
    "data/raw/PAD-UFES-20/metadata_split.csv"
)

IMAGE_ROOT = Path(
    "data/raw/PAD-UFES-20/images"
)

OUTPUT_DIR = Path(
    "models/gradcam"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("GRAD-CAM EXPLAINABILITY")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading model...")

model = DermatologyEfficientNet(
    num_classes=6,
    freeze_backbone=False,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print(
    "Loaded checkpoint:",
    CHECKPOINT_PATH
)

print(
    "Checkpoint epoch:",
    checkpoint.get("epoch", "unknown")
)

print(
    "Best validation Macro F1:",
    checkpoint.get("best_val_f1", "unknown")
)


# ============================================================
# FIND TEST IMAGE
# ============================================================

print()
print("Finding test image...")

df = pd.read_csv(METADATA_PATH)

test_df = df[
    df["split"] == "test"
].reset_index(drop=True)

# Use first test image
row = test_df.iloc[0]

img_id = row["img_id"]
actual_class = row["diagnostic"]

matches = list(
    IMAGE_ROOT.rglob(img_id)
)

if not matches:
    raise FileNotFoundError(
        f"Image not found: {img_id}"
    )

image_path = matches[0]

print("Image:", img_id)
print("Actual class:", actual_class)
print("Path:", image_path)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

image = Image.open(
    image_path
).convert("RGB")

original_image = np.array(
    image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )
)

rgb_image = (
    original_image.astype(np.float32)
    / 255.0
)

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

input_tensor = transform(
    image
).unsqueeze(0).to(device)


# ============================================================
# PREDICTION
# ============================================================

print()
print("Running prediction...")

with torch.no_grad():
    outputs = model(input_tensor)

probabilities = torch.softmax(
    outputs,
    dim=1
)

predicted_index = (
    probabilities.argmax(dim=1).item()
)

predicted_class = (
    CLASS_NAMES[predicted_index]
)

confidence = (
    probabilities[0, predicted_index]
    .item()
)

print(
    "Predicted class:",
    predicted_class
)

print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)


# ============================================================
# GRAD-CAM
# ============================================================

print()
print("Generating Grad-CAM...")

# Last convolutional feature layer
target_layers = [
    model.model.features[-1]
]

targets = [
    ClassifierOutputTarget(
        predicted_index
    )
]

with GradCAM(
    model=model,
    target_layers=target_layers,
) as cam:

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
    )

    grayscale_cam = (
        grayscale_cam[0]
    )


# ============================================================
# CREATE HEATMAP
# ============================================================

visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True,
)


# ============================================================
# SAVE RESULT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    OUTPUT_DIR
    / f"gradcam_{img_id}"
)

output_path = output_path.with_suffix(
    ".jpg"
)

cv2.imwrite(
    str(output_path),
    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    ),
)


# ============================================================
# SAVE ORIGINAL IMAGE TOO
# ============================================================

original_path = (
    OUTPUT_DIR
    / f"original_{img_id}"
)

original_path = original_path.with_suffix(
    ".jpg"
)

cv2.imwrite(
    str(original_path),
    cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2BGR
    ),
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("GRAD-CAM COMPLETE")
print("=" * 60)

print(
    "Actual class:",
    actual_class
)

print(
    "Predicted class:",
    predicted_class
)

print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)

print(
    "Original image:",
    original_path
)

print(
    "Grad-CAM image:",
    output_path
)