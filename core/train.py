"""Training script for SpecXNet using direct video loading."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project-root imports work when running:
#   python core/train.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataloader import VideoFramePairDataset, VideoSample, collect_video_samples, split_video_samples
from core.model import SpecXNet, SpecXNetConfig
from utils.prepare_data import cleanup_temp_frames


@dataclass
class TrainConfig:
    data_root: Path
    save_dir: Path
    real_dir: Path | None = None
    fake_dir: Path | None = None
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    pretrained: bool = True
    amp: bool = True
    frames_per_video: int = 16
    cleanup_temp: bool = True
    max_videos_per_class: int = 0
    # Regularization / stability
    dropout: float = 0.5
    label_smoothing: float = 0.08
    early_stop_patience: int = 0
    # ReduceLROnPlateau monitors val_loss (not val_acc).
    lr_plateau_patience: int = 2
    lr_plateau_factor: float = 0.5
    dfa_warmup_epochs: int = 3
    dfa_warmup_lambda: float = 0.15
    dfa_warmup_freq_floor: float = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.r_[np.inf, np.sort(np.unique(y_score))[::-1]]
    tpr = []
    fpr = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(np.int32)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tpr.append(tp / max(tp + fn, 1))
        fpr.append(fp / max(fp + tn, 1))
    return np.asarray(fpr), np.asarray(tpr), thresholds


def auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def plot_and_save_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fusion_weights: np.ndarray,
    save_dir: Path,
) -> dict[str, float]:
    save_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_score >= 0.5).astype(np.int32)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    fpr, tpr, _ = binary_roc_curve(y_true, y_score)
    auc = auc_trapezoid(fpr, tpr)

    # ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#d98ca3", linewidth=2, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curve.png", dpi=200)
    plt.close()

    # Confusion matrix
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int32)
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    ticks = [0, 1]
    labels = ["Real", "Fake"]
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="#222222")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # DFA weight histogram
    spatial_w = fusion_weights[:, 0]
    freq_w = fusion_weights[:, 1]
    plt.figure(figsize=(7, 4.5))
    plt.hist(spatial_w, bins=30, alpha=0.65, label="Spatial Weight", color="#77a6d4")
    plt.hist(freq_w, bins=30, alpha=0.65, label="Frequency Weight", color="#d98ca3")
    plt.xlabel("Weight")
    plt.ylabel("Count")
    plt.title("DFA Fusion Weight Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "dfa_weight_stats.png", dpi=200)
    plt.close()

    metrics = {
        "auc": auc,
        "accuracy": float(np.mean(y_pred == y_true)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "spatial_weight_mean": float(np.mean(spatial_w)),
        "frequency_weight_mean": float(np.mean(freq_w)),
    }
    with (save_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_epoch(
    model: SpecXNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    label_smoothing: float = 0.0,
    dfa_warmup_epochs: int = 0,
    dfa_warmup_lambda: float = 0.0,
    dfa_warmup_freq_floor: float = 0.2,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0
    seen_samples = 0
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_gates: list[np.ndarray] = []

    phase = "train" if training else "val"
    progress = tqdm(
        loader,
        desc=f"[{phase}] epoch {epoch_idx:03d}/{total_epochs:03d}",
        leave=False,
    )

    for rgb, fft, target in progress:
        rgb = rgb.to(device, non_blocking=True)
        fft = fft.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, gates = model(
                    rgb,
                    fft,
                    return_logits=True,
                    return_fusion_weights=True,
                )
                loss_target = target
                if training and label_smoothing > 0.0:
                    # Symmetric smoothing for binary BCE (reduces overconfidence / overfitting).
                    ls = float(label_smoothing)
                    loss_target = target * (1.0 - ls) + 0.5 * ls
                loss = criterion(logits, loss_target)
                if (
                    training
                    and dfa_warmup_epochs > 0
                    and dfa_warmup_lambda > 0.0
                    and epoch_idx <= dfa_warmup_epochs
                ):
                    w_freq = gates[:, 1]
                    deficit = F.relu(float(dfa_warmup_freq_floor) - w_freq).mean()
                    loss = loss + float(dfa_warmup_lambda) * deficit

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == target).sum().item()
        total += target.numel()
        batch_n = rgb.size(0)
        seen_samples += batch_n
        total_loss += loss.item() * batch_n
        all_targets.append(target.detach().cpu().numpy().reshape(-1))
        all_probs.append(probs.detach().cpu().numpy().reshape(-1))
        all_gates.append(gates.detach().cpu().numpy())

        running_loss = total_loss / max(seen_samples, 1)
        running_acc = correct / max(total, 1)
        progress.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = correct / max(total, 1)
    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.float32)
    y_prob = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
    fusion_weights = np.concatenate(all_gates) if all_gates else np.zeros((0, 2), dtype=np.float32)
    return avg_loss, acc, y_true, y_prob, fusion_weights


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.real_dir is not None or cfg.fake_dir is not None:
        samples: list[VideoSample] = []
        if cfg.real_dir is not None:
            real_samples = collect_video_samples(cfg.real_dir)
            # Force explicit label to avoid path-dependent inference mistakes.
            for s in real_samples:
                s.label = 0
            samples.extend(real_samples)
        if cfg.fake_dir is not None:
            fake_samples = collect_video_samples(cfg.fake_dir)
            for s in fake_samples:
                s.label = 1
            samples.extend(fake_samples)
    else:
        samples = collect_video_samples(cfg.data_root)

    if not samples:
        if cfg.real_dir is not None or cfg.fake_dir is not None:
            raise RuntimeError(
                f"No video samples found. real_dir={cfg.real_dir}, fake_dir={cfg.fake_dir}"
            )
        raise RuntimeError(f"No video samples found under {cfg.data_root}.")

    if cfg.max_videos_per_class > 0:
        by_label: dict[int, list[VideoSample]] = {0: [], 1: []}
        for s in samples:
            by_label.setdefault(int(s.label), []).append(s)
        rng = random.Random(cfg.seed)
        limited: list[VideoSample] = []
        for label in [0, 1]:
            group = by_label.get(label, [])
            rng.shuffle(group)
            limited.extend(group[: cfg.max_videos_per_class])
        samples = limited
        print(
            f"[INFO] max_videos_per_class={cfg.max_videos_per_class}, "
            f"using {len(samples)} videos total."
        )

    train_samples, val_samples = split_video_samples(samples, cfg.val_ratio, cfg.seed)
    if not train_samples or not val_samples:
        raise RuntimeError("Train/val split is empty. Add more data or reduce val_ratio.")

    train_ds = VideoFramePairDataset(
        train_samples,
        image_size=cfg.image_size,
        frames_per_video=cfg.frames_per_video,
        augment=True,
        seed=cfg.seed,
    )
    val_ds = VideoFramePairDataset(
        val_samples,
        image_size=cfg.image_size,
        frames_per_video=max(4, cfg.frames_per_video // 2),
        augment=False,
        seed=cfg.seed + 999,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpecXNet(
        SpecXNetConfig(pretrained=cfg.pretrained, dropout=cfg.dropout),
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = None
    if cfg.lr_plateau_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.lr_plateau_factor,
            patience=cfg.lr_plateau_patience,
        )
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_val_acc = 0.0
    use_amp = cfg.amp and device.type == "cuda"
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc, _, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            epoch_idx=epoch,
            total_epochs=cfg.epochs,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            label_smoothing=cfg.label_smoothing,
            dfa_warmup_epochs=cfg.dfa_warmup_epochs,
            dfa_warmup_lambda=cfg.dfa_warmup_lambda,
            dfa_warmup_freq_floor=cfg.dfa_warmup_freq_floor,
        )
        val_loss, val_acc, y_true, y_prob, gates = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch_idx=epoch,
            total_epochs=cfg.epochs,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            label_smoothing=0.0,
            dfa_warmup_epochs=0,
            dfa_warmup_lambda=0.0,
            dfa_warmup_freq_floor=cfg.dfa_warmup_freq_floor,
        )
        if scheduler is not None:
            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            lr_after = optimizer.param_groups[0]["lr"]
            if lr_after < lr_before:
                print(f"  -> ReduceLROnPlateau: lr {lr_before:g} -> {lr_after:g}")

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        latest_path = cfg.save_dir / "specxnet_latest.pth"
        torch.save(model.state_dict(), latest_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_path = cfg.save_dir / "specxnet_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved: {best_path} (val_acc={best_val_acc:.4f})")
            metrics = plot_and_save_metrics(
                y_true=y_true.astype(np.int32),
                y_score=y_prob.astype(np.float32),
                fusion_weights=gates.astype(np.float32),
                save_dir=cfg.save_dir,
            )
            print(
                f"  -> Metrics updated: AUC={metrics['auc']:.4f}, "
                f"SpatialW={metrics['spatial_weight_mean']:.4f}, "
                f"FreqW={metrics['frequency_weight_mean']:.4f}"
            )
        else:
            epochs_no_improve += 1
            if cfg.early_stop_patience > 0 and epochs_no_improve >= cfg.early_stop_patience:
                print(
                    f"[INFO] Early stopping: no val_acc improvement for "
                    f"{cfg.early_stop_patience} epoch(s). Best val_acc={best_val_acc:.4f}"
                )
                break

    if cfg.cleanup_temp:
        cleanup_temp_frames(PROJECT_ROOT / "data")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train SpecXNet with DFA fusion from videos.")
    parser.add_argument("--data_root", type=Path, default=Path("data/raw_videos"))
    parser.add_argument("--real_dir", type=Path, default=None, help="Path to real videos directory.")
    parser.add_argument("--fake_dir", type=Path, default=None, help="Path to fake videos directory.")
    parser.add_argument("--save_dir", type=Path, default=Path("weights"))
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--frames_per_video", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_cleanup_temp", action="store_true")
    parser.add_argument(
        "--max_videos_per_class",
        type=int,
        default=0,
        help="Limit number of videos per class (0 means use all).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout before classifier (higher = more regularization).",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.08,
        help="BCE label smoothing (0 disables). Typical: 0.05–0.1.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop if val_acc does not improve for N epochs (0 = disabled).",
    )
    parser.add_argument(
        "--lr_plateau_patience",
        type=int,
        default=2,
        help="Reduce LR when val_loss plateaus for N epochs (0 = disabled).",
    )
    parser.add_argument(
        "--lr_plateau_factor",
        type=float,
        default=0.5,
        help="LR multiplicative factor when plateau triggers.",
    )
    parser.add_argument(
        "--dfa_warmup_epochs",
        type=int,
        default=3,
        help="Epochs (1-based) with extra loss if DFA freq weight is below floor.",
    )
    parser.add_argument(
        "--dfa_warmup_lambda",
        type=float,
        default=0.15,
        help="Multiplier for freq-branch warm-up penalty (0 disables).",
    )
    parser.add_argument(
        "--dfa_warmup_freq_floor",
        type=float,
        default=0.2,
        help="Target minimum softmax weight for frequency stream during warm-up.",
    )
    args = parser.parse_args()

    return TrainConfig(
        data_root=args.data_root,
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        save_dir=args.save_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        frames_per_video=args.frames_per_video,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        amp=not args.no_amp,
        cleanup_temp=not args.no_cleanup_temp,
        max_videos_per_class=args.max_videos_per_class,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        early_stop_patience=args.early_stop_patience,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        dfa_warmup_epochs=args.dfa_warmup_epochs,
        dfa_warmup_lambda=args.dfa_warmup_lambda,
        dfa_warmup_freq_floor=args.dfa_warmup_freq_floor,
    )


if __name__ == "__main__":
    train(parse_args())
