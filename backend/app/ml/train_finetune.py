import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from backend.app.ml.dataloader import create_dataloaders
from backend.app.ml.model import DermatologyEfficientNet


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8
NUM_WORKERS = 0

EPOCHS = 15

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

EARLY_STOPPING_PATIENCE = 4

CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
]

BASELINE_CHECKPOINT = Path(
    "models/checkpoints/efficientnet_b0_best.pth"
)

CHECKPOINT_DIR = Path(
    "models/checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "efficientnet_b0_finetuned_best.pth"
)

HISTORY_PATH = (
    CHECKPOINT_DIR
    / "finetune_history.json"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("=" * 60)
    print("EXPERIMENT 2 - EFFICIENTNET-B0 FINE-TUNING")
    print("=" * 60)

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0)
                .total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    print()


    # ========================================================
    # DATA
    # ========================================================

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    (
        train_loader,
        val_loader,
        test_loader,
        class_weights,
    ) = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(
        "Train batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    print(
        "Test batches:",
        len(test_loader)
    )

    print()


    # ========================================================
    # MODEL
    # ========================================================

    print("=" * 60)
    print("CREATING FINE-TUNING MODEL")
    print("=" * 60)

    model = DermatologyEfficientNet(
        num_classes=len(CLASS_NAMES),
        freeze_backbone=False,
    )

    # Load the best model from Experiment 1.
    # This gives fine-tuning a task-trained starting point.
    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)


    # ========================================================
    # PARAMETER COUNT
    # ========================================================

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Total parameters:",
        total_parameters
    )

    print(
        "Trainable parameters:",
        trainable_parameters
    )

    print()


    # ========================================================
    # LOSS
    # ========================================================

    # Keep the same sampling strategy as Experiment 1.
    # Do not double-weight the minority classes yet.

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )


    # ========================================================
    # MIXED PRECISION
    # ========================================================

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )


    # ========================================================
    # TRAIN ONE EPOCH
    # ========================================================

    def train_one_epoch():

        model.train()

        running_loss = 0.0

        all_predictions = []
        all_labels = []

        for batch in train_loader:

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            labels = batch["label"].to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.detach()
                .cpu()
                .tolist()
            )

            all_labels.extend(
                labels.detach()
                .cpu()
                .tolist()
            )

        epoch_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        epoch_accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        epoch_f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )

        return (
            epoch_loss,
            epoch_accuracy,
            epoch_f1
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    def validate():

        model.eval()

        running_loss = 0.0

        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for batch in val_loader:

                images = batch["image"].to(
                    device,
                    non_blocking=True
                )

                labels = batch["label"].to(
                    device,
                    non_blocking=True
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

                running_loss += (
                    loss.item()
                    * images.size(0)
                )

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

        val_loss = (
            running_loss
            / len(val_loader.dataset)
        )

        val_accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        val_f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )

        return (
            val_loss,
            val_accuracy,
            val_f1
        )


    # ========================================================
    # TRAINING
    # ========================================================

    print("=" * 60)
    print("STARTING FINE-TUNING")
    print("=" * 60)

    history = []

    best_val_f1 = 0.0

    epochs_without_improvement = 0


    for epoch in range(1, EPOCHS + 1):

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )


        train_loss, train_accuracy, train_f1 = (
            train_one_epoch()
        )


        val_loss, val_accuracy, val_f1 = (
            validate()
        )


        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        scheduler.step(val_f1)


        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Train Macro F1: "
            f"{train_f1:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Val Macro F1: "
            f"{val_f1:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.7f}"
        )


        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_f1,
                "learning_rate": current_lr,
            }
        )


        with open(
            HISTORY_PATH,
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=4
            )


        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_val_f1,
                    "class_names": CLASS_NAMES,
                },
                BEST_MODEL_PATH,
            )

            print(
                "✓ New best fine-tuned model saved!"
            )

        else:

            epochs_without_improvement += 1

            print(
                "No improvement."
            )


        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break


    print()
    print("=" * 60)
    print("FINE-TUNING FINISHED")
    print("=" * 60)

    print(
        f"Best Validation Macro F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        "Best model:",
        BEST_MODEL_PATH
    )

    print(
        "Training history:",
        HISTORY_PATH
    )


if __name__ == "__main__":
    main()