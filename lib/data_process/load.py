import os
from pathlib import Path
from typing import Union, Tuple
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