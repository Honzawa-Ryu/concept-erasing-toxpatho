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
    n_slides: int = config.get("n_slides", 0)  # 0 (or falsy) means "use all slides"
    knn_patches_per_slide: int = config.get("knn_patches_per_slide", 50)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:              {run_dir}")
    logger.info(f"dataset_dir:          {dataset_dir}")
    logger.info(f"seed:                 {seed}")
    logger.info(f"n_slides:             {n_slides or 'all'}")
    logger.info(f"knn_patches_per_slide: {knn_patches_per_slide}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    # exp0010では「ぼやけスコア」を消してもhas_finding(病理所見)の判別性能は
    # ほぼ保たれることを確認した。今度は逆に、スライド間差（＝スライドID自体、
    # [[wsi-ad-batch-effect-finding]]と同じ意味でのバッチ効果）を消したときに
    # has_findingの判別性能がどう変わるかを見る。もしhas_finding分類が実は
    # 「どのスライドか」というショートカットに大きく依存しているなら、
    # スライドIDを消すとhas_finding性能も大きく落ちるはずである。
    from lib.data_process.load import load_memmaps_to_ram
    from lib.data_process.labels import load_finding_labels
    from lib.validate.batch_effect import (
        compute_eta_squared,
        compute_knn_accuracy,
        compute_effective_rank,
        compute_geometric_change,
        compute_logreg_probe,
    )
    from concept_erasure import LeaceFitter, QuadraticFitter
    from torch.nn.functional import one_hot

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device (for QuadraticEraser): {device}")

    X_all, slide_id_all = load_memmaps_to_ram(dataset_dir / "features_memmap_output", feature_dim=1024, dtype="float32")
    logger.info(f"Loaded full pool: X={X_all.shape}")

    if n_slides:
        rng_subset = np.random.default_rng(seed)
        all_slide_ids = sorted(set(slide_id_all.tolist()))
        chosen = rng_subset.choice(all_slide_ids, size=min(n_slides, len(all_slide_ids)), replace=False)
        mask = np.isin(slide_id_all, chosen)
        X, slide_id_per_patch = X_all[mask], slide_id_all[mask]
    else:
        X, slide_id_per_patch = X_all, slide_id_all

    slide_ids = sorted(set(slide_id_per_patch.tolist()))
    logger.info(f"Using {len(slide_ids)} slides, X={X.shape}")

    # ── has_finding(病理所見)ラベル: exp0010と同じ突合ロジック ───────────────
    finding_labels = load_finding_labels(
        dataset_dir / "raw_table" / "open_tggates_pathological_image.csv",
        dataset_dir / "raw_table" / "open_tggates_pathology.csv",
    )
    finding_labels = finding_labels.loc[finding_labels.index.isin(slide_ids)]
    missing = set(slide_ids) - set(finding_labels.index)
    if missing:
        raise RuntimeError(f"{len(missing)} slide(s) have no TG-GATEs label match: {sorted(missing)[:5]}")

    y_finding_slide = finding_labels.loc[slide_ids, "has_finding"].to_numpy().astype(int)
    logger.info(f"has_finding: {int(y_finding_slide.sum())} / {len(y_finding_slide)}")

    unique_labels, y_codes = np.unique(slide_id_per_patch, return_inverse=True)
    n_classes = len(unique_labels)
    logger.info(f"slide-id classes: {n_classes}")

    # スライドID(バッチ)のKNNプローブは最大で998クラスにもなるため、パッチ全量を
    # 無作為抽出すると偶然0件になるクラスが出てKFoldが壊れかねない。スライド
    # ごとに均等にknn_patches_per_slide枚だけ抽出し、各クラスの件数を揃える。
    rng = np.random.default_rng(seed)
    sub_idx_parts = []
    for sid in unique_labels:
        idx_this_slide = np.where(slide_id_per_patch == sid)[0]
        take = rng.choice(idx_this_slide, size=min(knn_patches_per_slide, len(idx_this_slide)), replace=False)
        sub_idx_parts.append(take)
    sub_idx = np.concatenate(sub_idx_parts)
    logger.info(f"KNN eval subsample: {len(sub_idx)} patches ({knn_patches_per_slide}/slide)")

    def mean_pool_by_slide(features: np.ndarray) -> np.ndarray:
        """パッチ特徴量をスライド単位で平均プーリングし、(n_slides, dim)にする。"""
        return np.stack([features[slide_id_per_patch == s].mean(axis=0) for s in slide_ids])

    def evaluate(X_: np.ndarray, tag: str) -> dict:
        eta = compute_eta_squared(X_, slide_id_per_patch)
        knn_slide_acc = compute_knn_accuracy(
            X_[sub_idx], y_codes[sub_idx], n_neighbors=15, n_splits=5, random_state=seed
        )
        effective_rank = compute_effective_rank(X_)
        finding_probe = compute_logreg_probe(mean_pool_by_slide(X_), y_finding_slide, n_splits=5, random_state=seed)
        logger.info(
            f"[{tag}] slide-id: eta_sq_mean={eta['eta_sq_mean']:.4f} "
            f"knn_acc={knn_slide_acc:.4f} (chance={1 / n_classes:.4f}) "
            f"effective_rank_ratio={effective_rank['effective_rank_ratio']:.4f} | "
            f"has_finding probe: balanced_acc={finding_probe['balanced_accuracy']:.4f} "
            f"roc_auc={finding_probe['roc_auc']:.4f}"
        )
        return {
            "compute_eta_squared": eta,
            "compute_knn_accuracy": knn_slide_acc,
            "compute_effective_rank": effective_rank,
            "finding_probe": finding_probe,
        }

    before_erasure = evaluate(X, "before")

    # ── LEACE (slide_idを one-hot のカテゴリ概念として消去; exp0005/0006と同じ手法) ──
    # one_hot(..., num_classes=n_classes)をパッチ全量に対して一括で作ると
    # (n_samples, n_classes)の密行列になり、n_classes~998では数GBに膨らんで
    # しまう。LeaceFitterのincremental update APIを使い、スライドごとに小さい
    # one-hotバッチを作っては捨てることでこれを避ける。
    leace_fitter = LeaceFitter(x_dim=X.shape[1], z_dim=n_classes, dtype=torch.float32)
    for class_idx, sid in enumerate(unique_labels):
        mask = slide_id_per_patch == sid
        x_batch = torch.from_numpy(X[mask]).float()
        z_batch = one_hot(
            torch.full((int(mask.sum()),), class_idx, dtype=torch.long), num_classes=n_classes
        ).float()
        leace_fitter.update(x_batch, z_batch)
    leace_eraser = leace_fitter.eraser

    X_tensor = torch.from_numpy(X).float()
    X_leace = leace_eraser(X_tensor).numpy()

    leace_result = evaluate(X_leace, "leace/after")
    leace_result["geometric_change"] = compute_geometric_change(X, X_leace)

    # ── QuadraticEraser (slide_idを離散クラスとして消去) ─────────────────────
    # QuadraticFitterのz(クラスラベル)は整数のインデックスでよく(one-hot不要)、
    # クラス数分の共分散(n_classes, 1024, 1024)に対するOTバリセンター計算だけが
    # 重い処理になる。GPU(バッチ化されたeigh)で実測したところ、998クラス
    # (998,000サンプル)でも fit+eraser 計算は約60秒・ピークメモリ約38GBで
    # 収まることを確認済みなので、ここでもGPUで計算する。
    z_tensor_gpu = torch.from_numpy(y_codes).long().to(device)
    X_tensor_gpu = torch.from_numpy(X).float().to(device)

    qe_fitter = QuadraticFitter.fit(X_tensor_gpu, z_tensor_gpu)
    qe_eraser = qe_fitter.eraser
    X_qe = qe_eraser(X_tensor_gpu, z_tensor_gpu).cpu().numpy()

    del X_tensor_gpu, z_tensor_gpu, qe_fitter, qe_eraser
    if device == "cuda":
        torch.cuda.empty_cache()

    qe_result = evaluate(X_qe, "quadratic/after")
    qe_result["geometric_change"] = compute_geometric_change(X, X_qe)

    results = {
        "n_slides": len(slide_ids),
        "n_classes": n_classes,
        "n_samples": int(X.shape[0]),
        "n_pos_slides": int(y_finding_slide.sum()),
        "n_neg_slides": int(len(y_finding_slide) - y_finding_slide.sum()),
        "knn_patches_per_slide": knn_patches_per_slide,
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