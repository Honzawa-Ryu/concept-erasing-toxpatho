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
    n_slides_per_class: int = config.get("n_slides_per_class", 60)
    n_bins: int = config.get("n_bins", 4)
    knn_eval_subsample: int = config.get("knn_eval_subsample", 10000)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:            {run_dir}")
    logger.info(f"dataset_dir:        {dataset_dir}")
    logger.info(f"seed:               {seed}")
    logger.info(f"n_slides_per_class: {n_slides_per_class}")
    logger.info(f"n_bins:             {n_bins}")
    logger.info(f"knn_eval_subsample: {knn_eval_subsample}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    # exp0008(LEACE)・exp0009(QuadraticEraser)は「ぼやけスコアの概念を消せたか」
    # しか見ていなかった。ここでは消去後のUNI特徴量が、ぼやけスコアとは無関係な
    # 「本物の」情報（Open TG-GATEsの病理所見の有無）をまだ持っているかを確認し、
    # 両手法とも「狙った概念だけを消していて、他の有用な情報を巻き込んで壊して
    # いないか」を検証する。
    #
    # 病理所見(has_finding)はスライド単位のラベルであり、1スライド内の全パッチが
    # 所見を示すわけではない（局所病変）ため、パッチ単位で直接プローブすると
    # ラベルノイズが大きすぎる。事前検証で実データを使い、
    #   - パッチ単位のKNN(スライドでグループ化してリークも防止): chance level
    #   - スライド単位で平均プーリングしKNNで分類: chance level
    #   - スライド単位で平均プーリングし正則化ロジスティック回帰で分類: balanced_accuracy=0.64, ROC-AUC=0.69
    # だったため、ここでは「パッチをスライド単位で平均プーリング→ロジスティック
    # 回帰プローブ」を使う（サンプル数(=スライド数) << 次元数(1024)の状況では、
    # KNNは次元の呪いで機能しないが、正則化された線形モデルは安定して働く）。
    from lib.data_process.labels import load_finding_labels
    from lib.data_process.load import load_blur_and_uni_features
    from lib.validate.batch_effect import (
        compute_effective_rank,
        compute_geometric_change,
        compute_knn_accuracy,
        compute_knn_regression_r2,
        compute_linear_regression_r2,
        compute_logreg_probe,
    )
    from concept_erasure import LeaceEraser, QuadraticEraser

    images_dir = dataset_dir / "memmap_output"
    features_dir = dataset_dir / "features_memmap_output"
    blur_output_dir = run_dir / "blur_score_output"

    # ── スライド部分集合の決定 ────────────────────────────────────────────────
    # has_findingの自然な発生率は約19%と低いので、そのまま先頭N枚を使うと
    # 陽性がほぼ0になる（exp0008/0009で実際に確認済み）。ここでは
    # has_finding True/Falseそれぞれからn_slides_per_class枚ずつ無作為抽出し、
    # 分類プローブが安定するようクラスバランスを揃える。
    image_ids = {d.name for d in images_dir.iterdir() if d.is_dir()}
    feature_ids = {d.name for d in features_dir.iterdir() if d.is_dir()}
    common_ids = image_ids & feature_ids

    finding_labels = load_finding_labels(
        dataset_dir / "raw_table" / "open_tggates_pathological_image.csv",
        dataset_dir / "raw_table" / "open_tggates_pathology.csv",
    )
    finding_labels = finding_labels.loc[finding_labels.index.isin(common_ids)]

    rng = np.random.default_rng(seed)
    pos_ids = finding_labels[finding_labels["has_finding"]].index.to_numpy()
    neg_ids = finding_labels[~finding_labels["has_finding"]].index.to_numpy()
    sel_pos = rng.choice(pos_ids, size=min(n_slides_per_class, len(pos_ids)), replace=False)
    sel_neg = rng.choice(neg_ids, size=min(n_slides_per_class, len(neg_ids)), replace=False)
    slide_ids = sorted(sel_pos.tolist() + sel_neg.tolist())
    logger.info(f"Subset: {len(slide_ids)} slides ({len(sel_pos)} has_finding / {len(sel_neg)} normal)")

    X, Z, slide_id_per_patch = load_blur_and_uni_features(images_dir, features_dir, blur_output_dir, slide_ids)
    logger.info(f"Loaded X={X.shape} Z={Z.shape}")

    y_finding_slide = finding_labels.loc[slide_ids, "has_finding"].to_numpy().astype(int)

    # KNN系のprobeはサンプル数の2乗近くコストがかかるため、パッチ全量ではなく
    # 固定シードで抽出した部分集合の上で評価する（消去前後で同じ部分集合を使う
    # ので比較の公平性は保たれる）。
    sub_idx = rng.choice(len(X), size=min(knn_eval_subsample, len(X)), replace=False)

    def mean_pool_by_slide(features: np.ndarray) -> np.ndarray:
        """パッチ特徴量をスライド単位で平均プーリングし、(n_slides, dim)にする。"""
        return np.stack([features[slide_id_per_patch == s].mean(axis=0) for s in slide_ids])

    def evaluate(X_: np.ndarray, tag: str) -> dict:
        knn_r2 = compute_knn_regression_r2(X_[sub_idx], Z[sub_idx], n_neighbors=15, n_splits=5, random_state=seed)
        linear_r2 = compute_linear_regression_r2(X_, Z, n_splits=5, random_state=seed)
        effective_rank = compute_effective_rank(X_)
        finding_probe = compute_logreg_probe(mean_pool_by_slide(X_), y_finding_slide, n_splits=5, random_state=seed)
        logger.info(
            f"[{tag}] blur: knn_r2={knn_r2:.4f} linear_r2={linear_r2:.4f} "
            f"effective_rank_ratio={effective_rank['effective_rank_ratio']:.4f} | "
            f"has_finding probe: balanced_acc={finding_probe['balanced_accuracy']:.4f} "
            f"roc_auc={finding_probe['roc_auc']:.4f}"
        )
        return {
            "compute_knn_regression_r2": knn_r2,
            "compute_linear_regression_r2": linear_r2,
            "compute_effective_rank": effective_rank,
            "finding_probe": finding_probe,
        }

    before_erasure = evaluate(X, "before")

    X_tensor = torch.from_numpy(X).float()

    # ── LEACE (連続値のぼやけスコアをそのまま消去; exp0008と同じ) ────────────
    leace_eraser = LeaceEraser.fit(X_tensor, torch.from_numpy(Z).float().unsqueeze(1))
    X_leace = leace_eraser(X_tensor).numpy()
    leace_result = evaluate(X_leace, "leace/after")
    leace_result["geometric_change"] = compute_geometric_change(X, X_leace)

    # ── QuadraticEraser (ぼやけスコアを分位点でn_bins個の離散クラスに変換して消去; exp0009と同じ) ──
    z_bins, bin_edges = pd.qcut(Z, q=n_bins, labels=False, retbins=True)
    z_bins = z_bins.astype(np.int64)
    missing_bins = set(range(n_bins)) - set(np.unique(z_bins).tolist())
    if missing_bins:
        raise RuntimeError(
            f"Quantile binning left classes {missing_bins} empty (n_bins={n_bins}); "
            f"reduce n_bins or check the blur score distribution."
        )
    z_bins_tensor = torch.from_numpy(z_bins).long()

    qe_eraser = QuadraticEraser.fit(X_tensor, z_bins_tensor)
    X_qe = qe_eraser(X_tensor, z_bins_tensor).numpy()
    qe_result = evaluate(X_qe, "quadratic/after")
    qe_result["geometric_change"] = compute_geometric_change(X, X_qe)
    qe_result["knn_bin_accuracy"] = compute_knn_accuracy(
        X_qe[sub_idx], z_bins[sub_idx], n_neighbors=15, n_splits=5, random_state=seed
    )
    qe_result["n_bins"] = n_bins
    qe_result["bin_edges"] = bin_edges.tolist()

    results = {
        "n_slides": len(slide_ids),
        "n_pos_slides": int(len(sel_pos)),
        "n_neg_slides": int(len(sel_neg)),
        "n_samples": int(X.shape[0]),
        "slide_ids": slide_ids,
        "before_erasure": before_erasure,
        "leace": leace_result,
        "quadratic": qe_result,
    }

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()