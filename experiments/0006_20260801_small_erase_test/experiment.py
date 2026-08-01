import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

# --- Deep learning (uncomment if needed) ---
import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset

# --- Visualization (uncomment if needed) ---
# import matplotlib.pyplot as plt
# import seaborn as sns


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(exp_dir: Path) -> dict:
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def compute_geometric_change(X: np.ndarray, X_erased: np.ndarray) -> dict:
    """Geometric evaluation of how much erasure deformed the representation.

    Written here for now; move to lib/validate/ once the interface settles.
    """
    diff = X_erased - X
    mse = float(np.mean(diff ** 2))
    mean_l2_distance = float(np.mean(np.linalg.norm(diff, axis=1)))

    x_norm = np.linalg.norm(X, axis=1)
    x_erased_norm = np.linalg.norm(X_erased, axis=1)
    denom = np.where(x_norm * x_erased_norm == 0, 1e-10, x_norm * x_erased_norm)
    cos_sim = np.sum(X * X_erased, axis=1) / denom

    total_var_before = np.sum(np.var(X, axis=0))
    total_var_after = np.sum(np.var(X_erased, axis=0))
    variance_ratio = float(total_var_after / total_var_before) if total_var_before > 0 else 0.0

    return {
        "mse": mse,
        "mean_l2_distance": mean_l2_distance,
        "cosine_similarity_mean": float(np.mean(cos_sim)),
        "cosine_similarity_std": float(np.std(cos_sim)),
        "variance_ratio": variance_ratio,
    }


def compute_mlp_probe_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: tuple = (128,),
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """Nonlinear probing: how well an MLP can still recover the erased concept.

    KNN (compute_knn_accuracy) is already a nonlinear probe; this adds a
    second model class as a cross-check. Written here for now; move to
    lib/validate/ once the interface settles.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        early_stopping=True,
        max_iter=200,
        random_state=random_state,
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(mlp, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    return float(np.mean(scores))


def parse_args() -> argparse.Namespace:
    """Define CLI args for all variable dimensions (used in GRID_ARGS / RUN_COMMAND)."""
    parser = argparse.ArgumentParser()
    # Add one argument per swept dimension (required=True).
    # These must match GRID_ARGS entries in run_slurm.sh.
    # Example:
    #   parser.add_argument("--model", required=True)
    #   parser.add_argument("--seed",  required=True)
    parser.add_argument("--config", required=True, help="Path to the config.yml file.")
    return parser.parse_args()


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()

    # Build variant_key from all variable dimensions so each config gets its own directory.
    # Include every arg that changes the result (model, seed, prompt, ...).
    # Example:
    #   model_short = args.model.replace("/", "-")
    #   variant_key = f"{model_short}__{args.seed}"
    variant_key = "default"

    # Initialize output directory (exits immediately if already completed).
    # Written on OUTPUT_ROOT (scratch) when set; scripts/slurm_entry.sh
    # rsyncs it back to project_root/outputs/ at job end.
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)
    n_slides: int = config.get("n_slides", 10)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")
    logger.info(f"n_slides:    {n_slides}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    # Smoke test combining experiment 0003 (batch-effect validation) and
    # 0005 (LEACE concept erasure) on a small slide subset, so the full
    # validate -> erase -> validate pipeline can be checked quickly.
    from lib.data_process.load import load_memmaps_to_ram
    from lib.validate.batch_effect import compute_eta_squared, compute_knn_accuracy
    from concept_erasure import LeaceEraser
    from torch.nn.functional import one_hot

    X_all, y_all = load_memmaps_to_ram(dataset_dir / "features_memmap_output", feature_dim=1024, dtype="float32")

    # Deterministic subset: first n_slides slide-ids in sorted order.
    slide_ids = np.unique(y_all)[:n_slides]
    mask = np.isin(y_all, slide_ids)
    X, y = X_all[mask], y_all[mask]
    logger.info(f"Subset: {len(slide_ids)} slides, {X.shape[0]} samples")

    # y is an array of slide-id strings (object dtype); torch can't convert
    # that directly, and sklearn's MLPClassifier chokes on string labels
    # internally when early_stopping=True, so label-encode to ints once and
    # reuse everywhere a numeric label is needed (one-hot, MLP probe).
    unique_labels, y_codes = np.unique(y, return_inverse=True)

    # ── Before erasure (0003-style batch-effect validation) ────────────────
    eta_before = compute_eta_squared(X, y)
    knn_before = compute_knn_accuracy(X, y, n_neighbors=15, n_splits=5, random_state=seed)
    mlp_before = compute_mlp_probe_accuracy(X, y_codes, random_state=seed)
    logger.info(
        f"Before erasure: eta_sq_mean={eta_before['eta_sq_mean']:.4f} "
        f"knn_acc={knn_before:.4f} mlp_acc={mlp_before:.4f}"
    )

    # ── Erasure (0005-style LEACE) ───────────────────────────────────────────
    X_tensor = torch.from_numpy(X).float()
    y_tensor = one_hot(torch.from_numpy(y_codes).long(), num_classes=len(unique_labels)).float()

    eraser = LeaceEraser.fit(X_tensor, y_tensor)
    X_erased = eraser(X_tensor).numpy()

    # ── After erasure (0003-style batch-effect validation) ──────────────────
    eta_after = compute_eta_squared(X_erased, y)
    knn_after = compute_knn_accuracy(X_erased, y, n_neighbors=15, n_splits=5, random_state=seed)
    mlp_after = compute_mlp_probe_accuracy(X_erased, y_codes, random_state=seed)
    logger.info(
        f"After erasure:  eta_sq_mean={eta_after['eta_sq_mean']:.4f} "
        f"knn_acc={knn_after:.4f} mlp_acc={mlp_after:.4f}"
    )

    # ── How much information erasure removed (geometric evaluation) ─────────
    geometric_change = compute_geometric_change(X, X_erased)
    logger.info(
        f"Geometric change: mse={geometric_change['mse']:.4f} "
        f"cos_sim={geometric_change['cosine_similarity_mean']:.4f} "
        f"variance_ratio={geometric_change['variance_ratio']:.4f}"
    )

    results = {
        "n_slides": int(len(slide_ids)),
        "n_samples": int(X.shape[0]),
        "before_erasure": {
            "compute_eta_squared": eta_before,
            "compute_knn_accuracy": knn_before,
            "compute_mlp_probe_accuracy": mlp_before,
        },
        "after_erasure": {
            "compute_eta_squared": eta_after,
            "compute_knn_accuracy": knn_after,
            "compute_mlp_probe_accuracy": mlp_after,
        },
        "geometric_change": geometric_change,
    }

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()