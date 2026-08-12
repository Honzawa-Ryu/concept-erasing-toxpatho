import json
import os
from pathlib import Path
from typing import List, Union, Tuple
import numpy as np

def load_memmaps_to_ram(
    base_dir: Union[str, Path],
    feature_dim: int = 1024,
    dtype: str = "float32"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    {slide_id}/features.dat の構造からデータを読み込み、
    RAM上で1つの巨大なNumPy配列として結合して返す関数。
    
    Returns
    -------
    X : np.ndarray, shape (total_samples, feature_dim)
        全スライドの特徴量が結合されたインメモリ配列。
    y : np.ndarray, shape (total_samples,)
        各行がどのスライド由来かを示すslide_id文字列の配列。
    """
    base_dir = Path(base_dir)
    element_size = np.dtype(dtype).itemsize
    
    X_parts = []
    y_parts = []
    total_samples = 0
    
    for slide_dir in sorted(base_dir.iterdir()):
        if not slide_dir.is_dir():
            continue
            
        dat_file = slide_dir / "features.dat"
        if not dat_file.exists():
            continue
            
        # ファイルサイズから行数を逆算
        file_size = os.path.getsize(dat_file)
        num_elements = file_size // element_size
        rows = num_elements // feature_dim
        
        if rows == 0:
            continue
            
        slide_id = slide_dir.name
        
        # memmapとして読み込み、np.arrayで明示的にメモリ(RAM)へコピーする
        m = np.memmap(dat_file, dtype=dtype, mode="r", shape=(rows, feature_dim))
        X_parts.append(np.array(m))
        y_parts.append(np.full(rows, slide_id, dtype=object))
        
        total_samples += rows
        
    if not X_parts:
        raise RuntimeError(f"No valid features.dat found under {base_dir}")
        
    print(f"Total slides loaded: {len(X_parts)}, Total samples: {total_samples}")
    
    # メモリ上で1つの配列に結合
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    return X, y


def load_blur_and_uni_features(
    images_dir: Union[str, Path],
    features_dir: Union[str, Path],
    blur_output_dir: Union[str, Path],
    slide_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    指定したスライド群について、UNI特徴量(features_memmap_output)とぼやけスコアを
    パッチ単位で対応づけて読み込む。

    blur_output_dir配下にまだぼやけスコアが無いスライドは
    lib.data_process.blur_score.process_single_slide_blur_score で計算してから
    読み込む（計算済みならスキップされる）。画像側(ぼやけスコア)と特徴量側(UNI)が
    同じパッチをサンプリングしているか(=indicesが一致するか)をスライドごとに
    確認し、ずれていれば例外を送出する。

    Returns
    -------
    X : np.ndarray, shape (total_samples, feature_dim)
        UNI特徴量。
    Z : np.ndarray, shape (total_samples,)
        ぼやけスコア。
    slide_id_per_patch : np.ndarray, shape (total_samples,)
        各行がどのスライド由来かを示すslide_id文字列の配列
        （スライド単位のラベルをパッチ単位にブロードキャストする用途などに使う）。
    """
    from lib.data_process.blur_score import process_single_slide_blur_score

    images_dir = Path(images_dir)
    features_dir = Path(features_dir)
    blur_output_dir = Path(blur_output_dir)

    X_parts, Z_parts, slide_id_parts = [], [], []
    for slide_id in slide_ids:
        process_single_slide_blur_score(
            memmap_dir=images_dir / slide_id,
            output_dir=blur_output_dir / slide_id,
        )
        blur_meta = json.loads((blur_output_dir / slide_id / "meta_blur.json").read_text())
        feat_meta = json.loads((features_dir / slide_id / "meta_features.json").read_text())

        if blur_meta["indices"] != feat_meta["indices"]:
            raise RuntimeError(
                f"Sample index mismatch between blur score and UNI feature for "
                f"slide {slide_id}; X and Z would not correspond patch-by-patch."
            )

        n = blur_meta["shape"][0]
        z = np.memmap(blur_output_dir / slide_id / "blur_scores.dat", dtype="float32", mode="r", shape=(n,))
        x = np.memmap(
            features_dir / slide_id / "features.dat",
            dtype=feat_meta["dtype"],
            mode="r",
            shape=tuple(feat_meta["shape"]),
        )

        X_parts.append(np.array(x))
        Z_parts.append(np.array(z))
        slide_id_parts.append(np.full(n, slide_id, dtype=object))

    X = np.concatenate(X_parts, axis=0)
    Z = np.concatenate(Z_parts, axis=0)
    slide_id_per_patch = np.concatenate(slide_id_parts, axis=0)

    return X, Z, slide_id_per_patch