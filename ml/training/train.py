"""Train Aperture's reference checkpoints: `policy.pt` (Run A) and `failure_head.pt` (Run B).

Run locally (CPU/MPS/CUDA) or in Colab — unlike the original notebook this has no Colab, Drive,
or `lerobot`-package dependency:

    python -m training.train --download --epochs 10 --device mps

Both runs import their model classes from `aperture.ml.model`, the same definitions the API
loads the checkpoints into. That import *is* the lockstep guarantee the notebook could only ask
for in a comment.

Run A  policy      real (frame, instruction) -> Gaussian over the action, Gaussian NLL loss.
Run B  failure head frozen pooled visual features -> perception / grounding / motor.

Both report held-out metrics on an episode-level split, so "it trained" is a measurement rather
than an assertion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.failure_surfaces import SURFACES, build_failure_dataset  # noqa: E402
from training.lerobot_data import LeRobotV3Dataset, download_dataset, pad_actions  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "ml" / "data" / "xarm_lift_medium"
DEFAULT_OUT = REPO_ROOT / "ml" / "models"

IMAGE_SIZE = 224


def pick_precision(requested: str, device: str) -> "torch.dtype | None":
    """Autocast dtype for this run, or None for full fp32.

    `bfloat16` is the default wherever the accelerator has it: it carries fp32's exponent
    range, so gradients cannot underflow and no `GradScaler` is needed, and on MPS it measured
    within 1% of fp16 (both ~1.6x faster than fp32). Parameters stay fp32 either way — autocast
    only lowers the precision of the compute — so checkpoints are unchanged and load into the
    fp32 inference path exactly as before.
    """
    if requested == "fp32":
        return None
    if requested in ("bf16", "fp16"):
        return torch.bfloat16 if requested == "bf16" else torch.float16
    return torch.bfloat16 if device in ("cuda", "mps") else None  # "auto"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- Run A: the policy ----------------------------------------------------------------------


class PolicyFrames(Dataset):
    """(frame, instruction, padded action) triples for the rows of one episode split.

    Frames stay at their native resolution here and are resized on-device per batch, which is
    both faster and exactly what the original training loop did.
    """

    def __init__(self, dataset: LeRobotV3Dataset, rows: np.ndarray, action_dim: int) -> None:
        self.frames = dataset.frames
        self.rows = rows
        self.actions = pad_actions(dataset.actions, action_dim)
        self.instructions = [dataset.episode_task[int(dataset.episode_index[r])] for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = int(self.rows[i])
        image = np.asarray(self.frames[row], dtype=np.float32) / 255.0  # HWC [0,1]
        return (
            torch.from_numpy(image).permute(2, 0, 1),
            self.instructions[i],
            torch.from_numpy(self.actions[row].copy()),
        )


def nll_action_loss(pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-pred_logvar)
    return (0.5 * inv_var * (target - pred_mean) ** 2 + 0.5 * pred_logvar).mean()


def _run_policy_epoch(model, loader, resize, device, opt=None, amp_dtype=None) -> float:
    training = opt is not None
    model.train(training)
    total, batches = 0.0, 0
    for image, instruction, action in loader:
        image = resize(image.to(device))
        action = action.to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                mean, logvar, _ = model(image, list(instruction))
            # The loss stays in fp32: the Gaussian NLL exponentiates a learned log-variance,
            # which is exactly the kind of term reduced precision handles worst.
            loss = nll_action_loss(mean.float(), logvar.float(), action)
        if training:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def train_policy(dataset: LeRobotV3Dataset, train_eps, val_eps, args, device: str, amp_dtype=None):
    from aperture.ml.model import ACTION_DIM, VISION_MODEL, AperturePolicy

    import timm

    model = AperturePolicy(action_dim=ACTION_DIM).to(device)
    # The shared class builds the ViT with `pretrained=False` (the checkpoint carries the weights
    # at inference time). Training does need the ImageNet initialisation, so load it in here
    # rather than forking the class definition.
    pretrained = timm.create_model(VISION_MODEL, pretrained=True, num_classes=0)
    model.vision.load_state_dict(pretrained.state_dict())

    if args.freeze_vision:
        # Fine-tuning the tower on one task's behaviour cloning collapses its general visual
        # features: the failure head, which reads the same tower, fell from 0.99 to 0.69
        # accuracy when trained on fine-tuned features instead of the ImageNet ones (see
        # ml/models/README.md). Aperture's policy exists to expose language->patch attention
        # and to answer counterfactual probes, not to be a state-of-the-art controller, so the
        # trade is firmly worth it: keep the general representation, train the fusion on top.
        for parameter in model.vision.parameters():
            parameter.requires_grad = False
        print("[run A] vision tower frozen (ImageNet features retained)", flush=True)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    for candidate in range(args.epochs - 1, -1, -1):
        path = ckpt_dir / f"policy_epoch{candidate}.pt"
        if path.exists():
            print(f"[run A] resuming from {path}", flush=True)
            model.load_state_dict(torch.load(path, map_location=device))
            start_epoch = candidate + 1
            break

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)

    train_loader = DataLoader(
        PolicyFrames(dataset, dataset.rows_for(train_eps), ACTION_DIM),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        PolicyFrames(dataset, dataset.rows_for(val_eps), ACTION_DIM),
        batch_size=args.batch_size, shuffle=False,
    )

    history = []
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        train_loss = _run_policy_epoch(model, train_loader, resize, device, opt, amp_dtype)
        val_loss = _run_policy_epoch(model, val_loader, resize, device, amp_dtype=amp_dtype)
        history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss})
        print(
            f"[run A] epoch {epoch} train_nll {train_loss:.4f} val_nll {val_loss:.4f} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
        torch.save(model.state_dict(), ckpt_dir / f"policy_epoch{epoch}.pt")

    return model, history


# --- Run B: the failure head ----------------------------------------------------------------


@torch.no_grad()
def pooled_features(model, images: np.ndarray, device: str, batch_size: int = 64, amp_dtype=None) -> torch.Tensor:
    """Frozen pooled visual embeddings for uint8 `[N,H,W,3]` images, matching inference exactly."""
    resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    model.eval()
    out = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size].astype(np.float32) / 255.0
        batch = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
        with torch.autocast(device, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            features = model.vision_features(resize(batch))
        # Back to fp32 before the head trains on them, so Run B is precision-independent.
        out.append(features.float().cpu())
    return torch.cat(out)


def train_failure_head(model, dataset: LeRobotV3Dataset, train_eps, val_eps, args, device: str, amp_dtype=None):
    from aperture.ml.model import FailureHead

    train_images, train_labels, train_variants = build_failure_dataset(
        dataset, train_eps, n_per_class=args.failure_samples, seed=args.seed
    )
    val_images, val_labels, val_variants = build_failure_dataset(
        dataset, val_eps, n_per_class=max(64, args.failure_samples // 8), seed=args.seed + 1
    )
    print(
        f"[run B] failure set: {len(train_images)} train / {len(val_images)} val frames "
        f"from {len(train_eps)}/{len(val_eps)} episodes",
        flush=True,
    )

    # The vision tower is frozen for Run B, so its features never change — extract once and the
    # head then trains in seconds instead of re-running the ViT every epoch.
    x_train = pooled_features(model, train_images, device, amp_dtype=amp_dtype)
    x_val = pooled_features(model, val_images, device, amp_dtype=amp_dtype)
    y_train = torch.from_numpy(train_labels)
    y_val = torch.from_numpy(val_labels)

    head = FailureHead(model.vision.num_features).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr)
    loss_fn = nn.CrossEntropyLoss()

    x_train_d, y_train_d = x_train.to(device), y_train.to(device)
    x_val_d, y_val_d = x_val.to(device), y_val.to(device)

    history = []
    for epoch in range(args.head_epochs):
        head.train()
        order = torch.randperm(len(x_train_d), device=device)
        total, batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            loss = loss_fn(head(x_train_d[idx]), y_train_d[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            batches += 1

        head.eval()
        with torch.no_grad():
            train_acc = float((head(x_train_d).argmax(-1) == y_train_d).float().mean())
            val_logits = head(x_val_d)
            val_acc = float((val_logits.argmax(-1) == y_val_d).float().mean())
        history.append(
            {"epoch": epoch, "loss": total / max(batches, 1), "train_acc": train_acc, "val_acc": val_acc}
        )
        print(
            f"[run B] epoch {epoch} loss {total / max(batches, 1):.4f} "
            f"train_acc {train_acc:.3f} val_acc {val_acc:.3f}",
            flush=True,
        )

    with torch.no_grad():
        predictions = head(x_val_d).argmax(-1).cpu().numpy()
    report = classification_report(val_labels, predictions, val_variants)
    return head, history, report


def leave_one_variant_out(model, dataset, train_eps, val_eps, args, device, amp_dtype) -> dict:
    """Does the head generalise to a corruption it never trained on?

    The headline accuracy is measured on the same six perception variants the head was trained
    with, so it cannot distinguish "understands that the visual channel failed" from "memorised
    these six corruptions". This holds each perception variant out of training entirely and
    scores only on it.

    The gap between this and the headline number is the honest measure of how much of the
    reported accuracy would survive a camera failing in a way the training set never contained.
    """
    from aperture.ml.model import FailureHead
    from training.failure_surfaces import PERCEPTION_VARIANTS, SURFACE_TO_IDX

    results: dict[str, float] = {}
    for held_out in PERCEPTION_VARIANTS:
        keep = [v for v in PERCEPTION_VARIANTS if v != held_out]

        train_images, train_labels, _ = build_failure_dataset(
            dataset, train_eps, n_per_class=args.failure_samples // 2,
            seed=args.seed, perception_variants=tuple(keep),
        )
        test_images, test_labels, test_variants = build_failure_dataset(
            dataset, val_eps, n_per_class=256, seed=args.seed + 2,
            perception_variants=(held_out,),
        )

        x_train = pooled_features(model, train_images, device, amp_dtype=amp_dtype).to(device)
        y_train = torch.from_numpy(train_labels).to(device)
        head = FailureHead(model.vision.num_features).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(args.head_epochs):
            head.train()
            order = torch.randperm(len(x_train), device=device)
            for start in range(0, len(order), args.batch_size):
                idx = order[start : start + args.batch_size]
                loss = loss_fn(head(x_train[idx]), y_train[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

        x_test = pooled_features(model, test_images, device, amp_dtype=amp_dtype).to(device)
        head.eval()
        with torch.no_grad():
            predictions = head(x_test).argmax(-1).cpu().numpy()
        # Score only the held-out perception rows: can it still tell the channel failed?
        mask = test_labels == SURFACE_TO_IDX["perception"]
        recall = float((predictions[mask] == test_labels[mask]).mean()) if mask.any() else 0.0
        results[held_out] = round(recall, 4)
        print(f"[run B] held-out variant {held_out:<14} perception recall {recall:.3f}", flush=True)

    return {
        "per_variant_recall": results,
        "mean_recall": round(float(np.mean(list(results.values()))), 4),
        "note": (
            "Perception recall on a corruption never seen in training. Compare against the "
            "headline per-class recall: the gap is memorisation rather than understanding."
        ),
    }


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, variants: list[str]) -> dict:
    confusion = np.zeros((len(SURFACES), len(SURFACES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    per_class = {}
    for i, surface in enumerate(SURFACES):
        support = int(confusion[i].sum())
        predicted = int(confusion[:, i].sum())
        recall = float(confusion[i, i] / support) if support else 0.0
        precision = float(confusion[i, i] / predicted) if predicted else 0.0
        per_class[surface] = {"precision": round(precision, 4), "recall": round(recall, 4), "support": support}

    by_variant: dict[str, dict] = {}
    for variant, t, p in zip(variants, y_true, y_pred):
        entry = by_variant.setdefault(variant, {"correct": 0, "total": 0})
        entry["total"] += 1
        entry["correct"] += int(t == p)
    for entry in by_variant.values():
        entry["accuracy"] = round(entry["correct"] / entry["total"], 4)

    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": SURFACES, "rows_true_cols_pred": confusion.tolist()},
        "per_variant_accuracy": by_variant,
    }


# --- Entry point ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--download", action="store_true", help="fetch the dataset first")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where policy.pt / failure_head.pt land")
    parser.add_argument("--checkpoint-dir", default=None, help="per-epoch checkpoints (default: <out>/checkpoints)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="autocast precision. 'auto' uses bfloat16 on a GPU (~1.6x faster, fp32 weights)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Run A epochs")
    parser.add_argument("--head-epochs", type=int, default=40, help="Run B epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--failure-samples", type=int, default=4000, help="Run B frames per surface")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="train on only the first N episodes — a smoke test of the whole script, not a real run",
    )
    parser.add_argument(
        "--freeze-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="freeze the ViT and train only the fusion + heads (default: on; see ml/models/README.md)",
    )
    parser.add_argument("--skip-policy", action="store_true", help="reuse <out>/policy.pt for Run B")
    parser.add_argument(
        "--skip-generalization",
        action="store_true",
        help="skip the leave-one-variant-out check (it retrains the head once per variant)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(out_dir / "checkpoints")

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    amp_dtype = pick_precision(args.precision, device)
    print(f"device: {device}  precision: {str(amp_dtype).replace('torch.', '') if amp_dtype else 'fp32'}")

    data_root = Path(args.data)
    if args.download:
        download_dataset(data_root)
    dataset = LeRobotV3Dataset.load(data_root)
    train_eps, val_eps = dataset.split_episodes(args.val_fraction, seed=args.seed)
    if args.max_episodes:
        # Keep the split episode-level — just take a prefix of each side.
        n_val = max(1, int(round(args.max_episodes * args.val_fraction)))
        train_eps = train_eps[: max(1, args.max_episodes - n_val)]
        val_eps = val_eps[:n_val]
        print(f"!! --max-episodes {args.max_episodes}: smoke run, not a publishable checkpoint")
    print(
        f"dataset: {len(dataset.frames)} frames, {len(dataset.episode_ids())} episodes "
        f"({len(train_eps)} train / {len(val_eps)} val)",
        flush=True,
    )

    if args.skip_policy:
        from aperture.ml.model import ACTION_DIM, AperturePolicy

        model = AperturePolicy(action_dim=ACTION_DIM).to(device)
        model.load_state_dict(torch.load(out_dir / "policy.pt", map_location=device))
        policy_history = []
    else:
        model, policy_history = train_policy(dataset, train_eps, val_eps, args, device, amp_dtype)
        torch.save(model.state_dict(), out_dir / "policy.pt")
        print(f"[run A] wrote {out_dir / 'policy.pt'}", flush=True)

    head, head_history, report = train_failure_head(
        model, dataset, train_eps, val_eps, args, device, amp_dtype
    )
    generalization = (
        leave_one_variant_out(model, dataset, train_eps, val_eps, args, device, amp_dtype)
        if not args.skip_generalization
        else None
    )
    torch.save(head.state_dict(), out_dir / "failure_head.pt")
    print(f"[run B] wrote {out_dir / 'failure_head.pt'}", flush=True)

    metrics = {
        "dataset": str(data_root),
        "device": device,
        "precision": str(amp_dtype).replace("torch.", "") if amp_dtype else "fp32",
        "episodes": {"train": len(train_eps), "val": len(val_eps)},
        "run_a_policy": {
            "epochs": args.epochs,
            "vision_frozen": bool(args.freeze_vision),
            "history": policy_history,
        },
        "run_b_failure_head": {
            "epochs": args.head_epochs,
            "samples_per_surface": args.failure_samples,
            "history": head_history,
            "validation": report,
            "generalization": generalization,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
