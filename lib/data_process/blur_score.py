import json
from pathlib import Path

import cv2
import numpy as np


def compute_laplacian_blur_score(image: np.ndarray) -> float:
    """1枚のRGBパッチに対し、ラプラシアンフィルタ適用後の分散でぼやけスコアを計算する。

    値が小さいほどエッジが少なく、ぼやけている（焦点が合っていない）パッチであることを示す。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process_single_slide_blur_score(
    memmap_dir: Path,
    output_dir: Path,
) -> None:
    """1スライド分のパッチMemmap(patches.dat)から各パッチのぼやけスコアを計算しMemmapに保存する"""
    src_meta_path = memmap_dir / "meta.json"
    src_memmap_file = memmap_dir / "patches.dat"

    if not src_meta_path.exists() or not src_memmap_file.exists():
        print(f"Skipping {memmap_dir.name} (source patches not found)")
        return

    meta_path = output_dir / "meta_blur.json"
    memmap_file = output_dir / "blur_scores.dat"

    # すでに処理が完了している場合はスキップ（レジューム機能）
    if meta_path.exists() and memmap_file.exists():
        print(f"Skipping {memmap_dir.name} (already processed)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(src_meta_path) as f:
        src_meta = json.load(f)

    shape = tuple(src_meta["shape"])
    dtype = src_meta["dtype"]
    num_samples = shape[0]

    # 1. ぼやけスコア計算
    patches = np.memmap(src_memmap_file, dtype=dtype, mode="r", shape=shape)
    blur_scores = np.empty(num_samples, dtype=np.float32)
    for idx in range(num_samples):
        blur_scores[idx] = compute_laplacian_blur_score(np.asarray(patches[idx]))

    # 2. メタデータ保存
    # 元のパッチMemmap(patches.dat)が使ったindices/coords/seedをそのまま引き継ぐことで、
    # features_memmap_output側と同じ並び順で対応づけられるようにする。
    meta = {
        "shape": [num_samples],
        "dtype": "float32",
        "method": "laplacian_variance",
        "seed": src_meta.get("seed"),
        "indices": src_meta.get("indices"),
        "coords": src_meta.get("coords"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    # 3. Memmap作成とスコア書き込み
    memmap_array = np.memmap(memmap_file, dtype=np.float32, mode="w+", shape=(num_samples,))
    memmap_array[:] = blur_scores
    memmap_array.flush()

    print(f"Saved {num_samples} blur scores for {memmap_dir.name}")


def batch_process_blur_score_directory(
    memmap_base_dir: str,
    output_base_dir: str,
) -> None:
    """指定されたディレクトリ内の全スライドのパッチMemmapについてぼやけスコアを計算する"""
    memmap_base_path = Path(memmap_base_dir)
    output_path = Path(output_base_dir)

    slide_dirs = sorted(d for d in memmap_base_path.iterdir() if d.is_dir())

    processed_count = 0
    for slide_dir in slide_dirs:
        slide_output_dir = output_path / slide_dir.name
        process_single_slide_blur_score(
            memmap_dir=slide_dir,
            output_dir=slide_output_dir,
        )
        processed_count += 1

    if processed_count == 0:
        raise RuntimeError(f"No slide directories found under {memmap_base_path}.")
