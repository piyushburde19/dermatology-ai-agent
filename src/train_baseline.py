"""
Baseline HAM10000 skin-lesion classifier (EfficientNet-B0).

Designed to run on a Kaggle or Colab GPU:
    python train_baseline.py --data_dir /kaggle/input/skin-cancer-mnist-ham10000 --epochs 8

Key points
- Splits by lesion_id (not by image), because HAM10000 has several images of the
  same lesion. An image-level split leaks near-duplicates into the test set and
  inflates accuracy.
- Uses square-root inverse-frequency class weights to handle class imbalance.
- Saves: best_model.pt, metrics.json, confusion_matrix.png, test_probs.npy,
  test_labels.npy (the last two are handy for confidence/human-review analysis).
"""
import argparse
import glob
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
MEL = CLASS_TO_IDX["mel"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_metadata(data_dir):
    csvs = glob.glob(os.path.join(data_dir, "**", "HAM10000_metadata*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found under {data_dir}")
    meta = pd.read_csv(csvs[0])

    jpgs = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in jpgs}
    meta["path"] = meta["image_id"].map(path_map)

    missing = int(meta["path"].isna().sum())
    if missing:
        print(f"Warning: {missing} images listed in metadata were not found; dropping them.")
    meta = meta.dropna(subset=["path"]).reset_index(drop=True)
    print(f"Loaded {len(meta)} images, {meta['lesion_id'].nunique()} unique lesions.")
    return meta


def split_by_lesion(meta, seed):
    """70/15/15 split with no lesion appearing in more than one split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_idx, tmp_idx = next(gss.split(meta, groups=meta["lesion_id"]))
    train, tmp = meta.iloc[tr_idx], meta.iloc[tmp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    v_idx, t_idx = next(gss2.split(tmp, groups=tmp["lesion_id"]))
    val, test = tmp.iloc[v_idx], tmp.iloc[t_idx]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class HAM(Dataset):
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = df["dx"].map(CLASS_TO_IDX).values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), int(self.labels[i])


def build_transforms(img_size):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(45),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            norm,
        ]
    )
    eval_tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])
    return train_tf, eval_tf


def build_model(pretrained):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_y = [], []
    use_amp = device.type == "cuda"
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x.to(device))
        all_probs.append(logits.float().softmax(1).cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_y)


def compute_metrics(probs, y):
    pred = probs.argmax(1)
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "melanoma_recall": float(recall_score(y == MEL, pred == MEL, zero_division=0)),
    }


def save_confusion_matrix(y, pred, path):
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true",
                    help="Use random init (only if the internet is off and weights cannot download).")
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    meta = load_metadata(args.data_dir)
    train_df, val_df, test_df = split_by_lesion(meta, args.seed)
    print(f"Split sizes -> train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")

    train_tf, eval_tf = build_transforms(args.img_size)
    make = lambda df, tf, shuffle: DataLoader(
        HAM(df, tf), batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    train_loader = make(train_df, train_tf, True)
    val_loader = make(val_df, eval_tf, False)
    test_loader = make(test_df, eval_tf, False)

    counts = np.bincount(train_df["dx"].map(CLASS_TO_IDX).values, minlength=len(CLASSES))
    weights = np.sqrt(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1, best_path = -1.0, os.path.join(args.out_dir, "best_model.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()

        val_probs, val_y = predict(model, val_loader, device)
        m = compute_metrics(val_probs, val_y)
        print(f"Epoch {epoch}/{args.epochs} | loss {running / len(train_df):.4f} | "
              f"val acc {m['accuracy']:.3f} | val macro-F1 {m['macro_f1']:.3f} | "
              f"val mel recall {m['melanoma_recall']:.3f}")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save(model.state_dict(), best_path)

    # Final evaluation on the held-out test split using the best validation checkpoint.
    model.load_state_dict(torch.load(best_path, map_location=device))
    probs, y = predict(model, test_loader, device)
    pred = probs.argmax(1)
    results = compute_metrics(probs, y)
    results["best_val_macro_f1"] = best_f1
    results["n_test"] = int(len(y))
    print("\nTEST RESULTS:", json.dumps(results, indent=2))
    print(classification_report(y, pred, labels=list(range(len(CLASSES))),
                                target_names=CLASSES, zero_division=0))

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out_dir, "classes.json"), "w") as f:
        json.dump(CLASSES, f)
    np.save(os.path.join(args.out_dir, "test_probs.npy"), probs)
    np.save(os.path.join(args.out_dir, "test_labels.npy"), y)
    save_confusion_matrix(y, pred, os.path.join(args.out_dir, "confusion_matrix.png"))
    print(f"\nSaved outputs to {args.out_dir}/")


if __name__ == "__main__":
    main()
