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
    # exp0007（ぼやけスコア）の概念を、exp0005/0006と同じLEACEでUNI特徴量から
    # 消去できるか試す小規模テスト。スライドIDのようなカテゴリ概念と違い、
    # ぼやけスコアは連続値なのでone-hot化はせず、そのままZとしてLEACEに渡す。
    from lib.data_process.load import load_blur_and_uni_features
    from lib.validate.batch_effect import (
        compute_geometric_change,
        compute_knn_regression_r2,
        compute_linear_regression_r2,
    )
    from concept_erasure import LeaceEraser

    images_dir = dataset_dir / "memmap_output"
    features_dir = dataset_dir / "features_memmap_output"
    blur_output_dir = run_dir / "blur_score_output"

    # Deterministic subset: 両方のソースに存在するスライドIDのうち、
    # ソート順で先頭からn_slides枚だけ使う（0006と同じ考え方）。
    image_ids = {d.name for d in images_dir.iterdir() if d.is_dir()}
    feature_ids = {d.name for d in features_dir.iterdir() if d.is_dir()}
    slide_ids = sorted(image_ids & feature_ids)[:n_slides]
    logger.info(f"Subset: {len(slide_ids)} slides -> {slide_ids}")

    X, Z = load_blur_and_uni_features(images_dir, features_dir, blur_output_dir, slide_ids)
    logger.info(f"Loaded X={X.shape} Z={Z.shape}")

    # ── Before erasure ───────────────────────────────────────────────────────
    # KNN(非線形)と線形回帰の両方でプローブする。LEACEは線形の予測可能性しか
    # 消去を保証しないため、線形R²がほぼ0まで落ちるかどうかが erasure が
    # 意図通り効いたかの主な判定基準になる。
    knn_r2_before = compute_knn_regression_r2(X, Z, n_neighbors=15, n_splits=5, random_state=seed)
    linear_r2_before = compute_linear_regression_r2(X, Z, n_splits=5, random_state=seed)
    logger.info(f"Before erasure: knn_r2={knn_r2_before:.4f} linear_r2={linear_r2_before:.4f}")

    # ── Erasure (LEACE, continuous concept) ──────────────────────────────────
    X_tensor = torch.from_numpy(X).float()
    Z_tensor = torch.from_numpy(Z).float().unsqueeze(1)  # (n, 1)

    eraser = LeaceEraser.fit(X_tensor, Z_tensor)
    X_erased = eraser(X_tensor).numpy()

    # ── After erasure ────────────────────────────────────────────────────────
    knn_r2_after = compute_knn_regression_r2(X_erased, Z, n_neighbors=15, n_splits=5, random_state=seed)
    linear_r2_after = compute_linear_regression_r2(X_erased, Z, n_splits=5, random_state=seed)
    logger.info(f"After erasure:  knn_r2={knn_r2_after:.4f} linear_r2={linear_r2_after:.4f}")

    # ── How much information erasure removed (geometric evaluation) ─────────
    geometric_change = compute_geometric_change(X, X_erased)
    logger.info(
        f"Geometric change: mse={geometric_change['mse']:.4f} "
        f"cos_sim={geometric_change['cosine_similarity_mean']:.4f} "
        f"variance_ratio={geometric_change['variance_ratio']:.4f}"
    )

    results = {
        "n_slides": len(slide_ids),
        "n_samples": int(X.shape[0]),
        "slide_ids": slide_ids,
        "before_erasure": {
            "compute_knn_regression_r2": knn_r2_before,
            "compute_linear_regression_r2": linear_r2_before,
        },
        "after_erasure": {
            "compute_knn_regression_r2": knn_r2_after,
            "compute_linear_regression_r2": linear_r2_after,
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