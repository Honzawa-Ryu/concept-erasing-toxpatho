import json
from pathlib import Path
import h5py
import numpy as np
import openslide

def process_single_slide(
    h5_path: Path,
    svs_path: Path,
    output_dir: Path,
    num_samples: int,
    patch_size: int = 224,
    level: int = 0,
    seed: int = 42
):
    """1つのSVS/H5ペアからMemmapを作成する"""
    meta_path = output_dir / "meta.json"
    memmap_file = output_dir / "patches.dat"

    # すでに処理が完了している場合はスキップ（レジューム機能）
    if meta_path.exists() and memmap_file.exists():
        print(f"Skipping {svs_path.stem} (already processed)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. サンプリング
    with h5py.File(h5_path, "r") as src:
        coords = src["coords"][:]
        total_patches = len(coords)

        if num_samples >= total_patches:
            sampled_coords = coords
        else:
            indices = rng.choice(total_patches, size=num_samples, replace=False)
            indices.sort()
            sampled_coords = coords[indices]

    actual_num_samples = len(sampled_coords)
    shape = (actual_num_samples, patch_size, patch_size, 3)
    dtype = np.uint8

    # 2. メタデータ保存
    with open(meta_path, "w") as f:
        json.dump({"shape": shape, "dtype": np.dtype(dtype).name}, f, indent=4)

    # 3. Memmap作成とパッチ書き込み
    memmap_array = np.memmap(memmap_file, dtype=dtype, mode="w+", shape=shape)
    slide = openslide.OpenSlide(str(svs_path))

    try:
        for idx, coord in enumerate(sampled_coords):
            x, y = int(coord[0]), int(coord[1])
            patch_img = slide.read_region(
                (x, y), level, (patch_size, patch_size)
            ).convert("RGB")

            memmap_array[idx] = np.array(patch_img, dtype=dtype)
            
            if (idx + 1) % 500 == 0:
                memmap_array.flush()
        memmap_array.flush()
    finally:
        slide.close()

    print(f"Saved {actual_num_samples} patches for {svs_path.stem}")


def batch_process_directory(
    h5_dir: str,
    svs_dir: str,
    output_base_dir: str,
    num_samples_per_slide: int,
    patch_size: int = 224,
    level: int = 0
):
    """
    指定された構造に基づき、H5ファイルとSVSファイルを紐づけて処理する
    """
    h5_path = Path(h5_dir)
    svs_path = Path(svs_dir)
    output_path = Path(output_base_dir)

    # H5ファイルを基準に探索 ("*_patches.h5")
    h5_files = list(h5_path.glob("*_patches.h5"))

    processed_count = 0

    for h5_file in h5_files:
        # "2656_patches.h5" から "2656" を抽出
        slide_id = h5_file.name.replace("_patches.h5", "")

        # 対応するSVSファイルのパスを構築
        svs_file = svs_path / f"{slide_id}.svs"

        if not svs_file.exists():
            print(f"Warning: SVS file {svs_file.name} not found. Skipping.")
            continue

        # 出力ディレクトリはID名 ("2656") にする
        slide_output_dir = output_path / slide_id

        process_single_slide(
            h5_path=h5_file,
            svs_path=svs_file,
            output_dir=slide_output_dir,
            num_samples=num_samples_per_slide,
            patch_size=patch_size,
            level=level
        )
        processed_count += 1

    # 1枚も処理できなかった場合、そのまま黙って完了扱いにされる（＝空の結果が
    # completion.json に「正式な完了」として残り、以後の再実行がガードで永久に
    # スキップされ続ける）のを防ぐため、ここで明示的に失敗させる。
    if processed_count == 0:
        raise RuntimeError(
            f"No slides were processed (found {len(h5_files)} H5 file(s), "
            f"0 matched an SVS file under {svs_path})."
        )

# 実行例
if __name__ == "__main__":
    H5_DIR = "/workspace/andre01/honzawa/02-playground/concept-erasing-toxpatho/data/trident_processed/20x_224px_0px_overlap/patches"
    SVS_DIR = "/workspace/andre01/honzawa/02-playground/concept-erasing-toxpatho/data/raw_slide"
    OUTPUT_DIR = "/workspace/andre01/honzawa/02-playground/concept-erasing-toxpatho/data/memmap_output"
    
    batch_process_directory(
        h5_dir=H5_DIR,
        svs_dir=SVS_DIR,
        output_base_dir=OUTPUT_DIR,
        num_samples_per_slide=1000, # 例: 各スライドから1000枚サンプリング
        patch_size=224,
        level=0
    )

def process_single_slide_feature(
    feature_h5_path: Path,
    output_dir: Path,
    num_samples: int,
    feature_key: str = "features",
    seed: int = 42
):
    """1つの特徴量H5ファイルからMemmapを作成する"""
    meta_path = output_dir / "meta_features.json"
    memmap_file = output_dir / "features.dat"

    # すでに処理が完了している場合はスキップ（レジューム機能）
    if meta_path.exists() and memmap_file.exists():
        print(f"Skipping {feature_h5_path.stem} (already processed)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. サンプリング
    with h5py.File(feature_h5_path, "r") as src:
        features = src[feature_key][:]
        total_features = len(features)

        if num_samples >= total_features:
            sampled_features = features
        else:
            indices = rng.choice(total_features, size=num_samples, replace=False)
            indices.sort()
            sampled_features = features[indices]

    actual_num_samples = len(sampled_features)
    shape = sampled_features.shape
    dtype = sampled_features.dtype

    # 2. メタデータ保存
    with open(meta_path, "w") as f:
        json.dump({"shape": shape, "dtype": str(dtype)}, f, indent=4)

    # 3. Memmap作成と特徴量書き込み
    memmap_array = np.memmap(memmap_file, dtype=dtype, mode="w+", shape=shape)
    memmap_array[:] = sampled_features[:]
    memmap_array.flush()

    print(f"Saved {actual_num_samples} features for {feature_h5_path.stem}")

def batch_process_feature_directory(
    feature_h5_dir: str,
    output_base_dir: str,
    num_samples_per_file: int,
    feature_key: str = "features"
):
    """
    指定されたディレクトリ内の特徴量H5ファイルを処理する
    """
    feature_h5_path = Path(feature_h5_dir)
    output_path = Path(output_base_dir)

    # H5ファイルを探索（trident出力は "{slide_id}.h5" 形式で接尾辞を持たない）
    feature_h5_files = list(feature_h5_path.glob("*.h5"))

    processed_count = 0

    for feature_h5_file in feature_h5_files:
        # "2656.h5" から "2656" を抽出
        file_id = feature_h5_file.stem

        # 出力ディレクトリはID名 ("2656") にする
        file_output_dir = output_path / file_id

        process_single_slide_feature(
            feature_h5_path=feature_h5_file,
            output_dir=file_output_dir,
            num_samples=num_samples_per_file,
            feature_key=feature_key
        )
        processed_count += 1

    if processed_count == 0:
        raise RuntimeError(
            f"No feature files were processed (found {len(feature_h5_files)} H5 file(s))."
        )