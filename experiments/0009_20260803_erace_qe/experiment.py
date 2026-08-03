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
    n_bins: int = config.get("n_bins", 4)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")
    logger.info(f"n_slides:    {n_slides}")
    logger.info(f"n_bins:      {n_bins}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    # exp0008はLEACE(線形消去)で試したが、KNN(非線形)プローブのR²があまり
    # 下がらなかった（線形の手がかりは消えても、分散などの2次以上の構造に
    # 手がかりが残っていた可能性）。QuadraticEraser は概念ごとの共分散を
    # 最適輸送(OT)でバリセンターに揃えることで、平均だけでなく分散・共分散
    # 構造まで消す非線形な手法。ただし離散クラスラベルが必要（かつ適用時にも
    # ラベルが必要）なので、連続値のぼやけスコアを分位点でn_bins個のクラスに
    # ビン分割してから使う。
    from lib.data_process.load import load_blur_and_uni_features
    from lib.validate.batch_effect import (
        compute_geometric_change,
        compute_knn_accuracy,
        compute_knn_regression_r2,
        compute_linear_regression_r2,
    )
    from concept_erasure import QuadraticEraser

    images_dir = dataset_dir / "memmap_output"
    features_dir = dataset_dir / "features_memmap_output"
    blur_output_dir = run_dir / "blur_score_output"

    # Deterministic subset: 両方のソースに存在するスライドIDのうち、
    # ソート順で先頭からn_slides枚だけ使う（0008と同じ考え方）。
    image_ids = {d.name for d in images_dir.iterdir() if d.is_dir()}
    feature_ids = {d.name for d in features_dir.iterdir() if d.is_dir()}
    slide_ids = sorted(image_ids & feature_ids)[:n_slides]
    logger.info(f"Subset: {len(slide_ids)} slides -> {slide_ids}")

    X, Z = load_blur_and_uni_features(images_dir, features_dir, blur_output_dir, slide_ids)
    logger.info(f"Loaded X={X.shape} Z={Z.shape}")

    # ── 連続値のぼやけスコアを分位点でn_bins個の離散クラスに変換 ─────────────
    z_bins, bin_edges = pd.qcut(Z, q=n_bins, labels=False, retbins=True)
    z_bins = z_bins.astype(np.int64)
    # QuadraticFitterはz.max()+1個のクラスを0始まりの連続整数と仮定して
    # 内部の統計量を配列に直接インデックスするため、空のクラスがあると
    # (OTバリセンター計算が退化して)壊れる。分位点ビニングなら通常は
    # 全クラスが必ず埋まるはずだが、念のため確認しておく。
    missing_bins = set(range(n_bins)) - set(np.unique(z_bins).tolist())
    if missing_bins:
        raise RuntimeError(
            f"Quantile binning left classes {missing_bins} empty (n_bins={n_bins}); "
            f"reduce n_bins or check the blur score distribution."
        )
    logger.info(f"Bin edges: {bin_edges.tolist()}")

    # ── Before erasure ───────────────────────────────────────────────────────
    # exp0008と同じ2種類のプローブ(連続値のぼやけスコアに対するKNN/線形回帰R²)に
    # 加えて、QuadraticEraserが直接ターゲットにしているビン(z_bins)自体を
    # KNN分類でどれだけ当てられるかも見る。
    knn_r2_before = compute_knn_regression_r2(X, Z, n_neighbors=15, n_splits=5, random_state=seed)
    linear_r2_before = compute_linear_regression_r2(X, Z, n_splits=5, random_state=seed)
    knn_bin_acc_before = compute_knn_accuracy(X, z_bins, n_neighbors=15, n_splits=5, random_state=seed)
    logger.info(
        f"Before erasure: knn_r2={knn_r2_before:.4f} linear_r2={linear_r2_before:.4f} "
        f"knn_bin_acc={knn_bin_acc_before:.4f} (chance={1 / n_bins:.4f})"
    )

    # ── Erasure (QuadraticEraser, 離散ビン) ──────────────────────────────────
    X_tensor = torch.from_numpy(X).float()
    z_tensor = torch.from_numpy(z_bins).long()

    eraser = QuadraticEraser.fit(X_tensor, z_tensor)
    X_erased = eraser(X_tensor, z_tensor).numpy()

    # ── After erasure ────────────────────────────────────────────────────────
    knn_r2_after = compute_knn_regression_r2(X_erased, Z, n_neighbors=15, n_splits=5, random_state=seed)
    linear_r2_after = compute_linear_regression_r2(X_erased, Z, n_splits=5, random_state=seed)
    knn_bin_acc_after = compute_knn_accuracy(X_erased, z_bins, n_neighbors=15, n_splits=5, random_state=seed)
    logger.info(
        f"After erasure:  knn_r2={knn_r2_after:.4f} linear_r2={linear_r2_after:.4f} "
        f"knn_bin_acc={knn_bin_acc_after:.4f}"
    )

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
        "n_bins": n_bins,
        "bin_edges": bin_edges.tolist(),
        "slide_ids": slide_ids,
        "before_erasure": {
            "compute_knn_regression_r2": knn_r2_before,
            "compute_linear_regression_r2": linear_r2_before,
            "compute_knn_bin_accuracy": knn_bin_acc_before,
        },
        "after_erasure": {
            "compute_knn_regression_r2": knn_r2_after,
            "compute_linear_regression_r2": linear_r2_after,
            "compute_knn_bin_accuracy": knn_bin_acc_after,
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