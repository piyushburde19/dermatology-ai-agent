import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from backend.app.ml.dataloader import create_dataloaders
from backend.app.ml.model import DermatologyEfficientNet


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8

CHECKPOINT_PATH = Path(
    "models/checkpoints/efficientnet_b0_weighted_best.pth"
)

RESULTS_DIR = Path(
    "models/evaluation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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
print("TEST SET EVALUATION")
print("=" * 60)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# DATA
# ============================================================

_, _, test_loader, _ = create_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=0,
)

print("Test images:", len(test_loader.dataset))
print("Test batches:", len(test_loader))
print()


# ============================================================
# MODEL
# ============================================================

model = DermatologyEfficientNet(
    num_classes=len(CLASS_NAMES)
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print(
    "Loaded checkpoint from epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation Macro F1:",
    f"{checkpoint['best_val_f1']:.4f}"
)

print()


# ============================================================
# INFERENCE
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )


# ============================================================
# METRICS
# ============================================================

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

test_macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

matrix = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Macro F1: {test_macro_f1:.4f}"
)

print()

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("Rows = Actual")
print("Columns = Predicted")
print()

print("Classes:", CLASS_NAMES)

print()

print(matrix)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "checkpoint_epoch": int(
        checkpoint["epoch"]
    ),
    "validation_macro_f1": float(
        checkpoint["best_val_f1"]
    ),
    "test_accuracy": float(
        test_accuracy
    ),
    "test_macro_f1": float(
        test_macro_f1
    ),
    "class_names": CLASS_NAMES,
    "confusion_matrix": matrix.tolist(),
    "classification_report": classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    ),
}


results_path = (
    RESULTS_DIR / "test_results.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print()

print(
    "Results saved to:",
    results_path
)