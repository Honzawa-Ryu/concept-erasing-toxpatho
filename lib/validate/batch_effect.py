from pathlib import Path
from typing import Dict, Tuple, Union

import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def compute_eta_squared(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    潜在表現の各次元における効果量（η²）を計算する関数。

    Eta^2 = SS_effect / SS_total

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。
    y : np.ndarray
        グループラベルのデータ。

    Returns
    -------
    Dict[str, float]
        平均値、中央値、および最大値のη²を含む辞書。
    """
    # Xの形状を取得
    # タプルはこうやってアンパックできる
    N, D = X.shape
    # ユニークなラベルとそのカウントを取得
    # np.uniqueは、配列内のユニークな要素を返す関数で、return_counts=Trueを指定すると、それぞれのユニークな要素の出現回数も返す
    unique_labels, counts = np.unique(y, return_counts=True)

    # ラベルによらない総平方和（SS_total）を計算
    grand_mean = np.mean(X, axis=0)
    
    # 全体の平方和を計算
    ss_total = np.sum((X - grand_mean) ** 2, axis=0)

    # グループごとの平均を計算
    group_means = np.zeros((len(unique_labels), D))
    # グループごとの平均を計算するために、各ラベルに対してXの対応する行を抽出し、その平均を計算
    for i, label in enumerate(unique_labels):
        group_means[i] = np.mean(X[y == label], axis=0)
    
    # グループ間平方和（SS_between）を計算
    ss_between = np.sum(counts[:, np.newaxis] * (group_means - grand_mean) ** 2, axis=0)

    # ゼロ除算を避けるために、ss_totalがゼロの場合は小さな値に置き換える
    ss_total = np.where(ss_total == 0, 1e-10, ss_total)  # Avoid division by zero

    # η²を計算
    eta_sq_per_dim = ss_between / ss_total

    return {
        "eta_sq_mean": float(np.mean(eta_sq_per_dim)),
        "eta_sq_median": float(np.median(eta_sq_per_dim)),
        "eta_sq_max": float(np.max(eta_sq_per_dim)),
        "eta_sq_var": float(np.var(eta_sq_per_dim))
    }


def compute_knn_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 15,
    n_splits: int = 5,
    random_state: int = 42
    ) -> float:
    """
    stratified K-Fold クロスバリデーションを使用して、潜在表現のKNN分類器の精度を計算する関数。

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。
    y : np.ndarray
        グループラベルのデータ。
    n_neighbors : int, optional
        KNN分類器の近傍数（デフォルトは15）。
    n_splits : int, optional
        クロスバリデーションの分割数（デフォルトは5）。
    random_state : int, optional
        乱数シード（デフォルトは42）。

    Returns
    -------
    float
        KNN分類器の平均精度。
    """
    # n_jobs=-1 is set on cross_val_score below, which already parallelizes
    # across folds; parallelizing here too would oversubscribe CPUs/memory.
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean', n_jobs=1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(knn, X, y, cv=skf, scoring='accuracy', n_jobs=-1)

    return float(np.mean(scores))


def compute_knn_regression_r2(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 15,
    n_splits: int = 5,
    random_state: int = 42
    ) -> float:
    """
    K-Fold クロスバリデーションを使用して、連続値の概念(y)をKNN回帰でどれだけ
    復元できるかをR²で評価する関数。compute_knn_accuracyの回帰版（yが
    スライドIDのような離散ラベルではなく、ぼやけスコアのような連続値の場合に使う）。

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。
    y : np.ndarray
        連続値の概念ラベル（例: ぼやけスコア）。
    n_neighbors : int, optional
        KNN回帰器の近傍数（デフォルトは15）。
    n_splits : int, optional
        クロスバリデーションの分割数（デフォルトは5）。
    random_state : int, optional
        乱数シード（デフォルトは42）。

    Returns
    -------
    float
        KNN回帰器の平均R²スコア。
    """
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric='euclidean', n_jobs=1)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(knn, X, y, cv=kf, scoring='r2', n_jobs=-1)

    return float(np.mean(scores))


def compute_linear_regression_r2(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
    ) -> float:
    """
    K-Fold クロスバリデーションを使用して、連続値の概念(y)を線形回帰でどれだけ
    復元できるかをR²で評価する関数。

    LEACEは線形の予測可能性しか消去を保証しないため、compute_knn_regression_r2
    （非線形プローブ）だけでは「線形の手がかりが消えたか」が分からない。この関数を
    ペアで使うことで、消去前後の線形R²がほぼ0まで落ちているかを確認できる。

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。
    y : np.ndarray
        連続値の概念ラベル（例: ぼやけスコア）。
    n_splits : int, optional
        クロスバリデーションの分割数（デフォルトは5）。
    random_state : int, optional
        乱数シード（デフォルトは42）。

    Returns
    -------
    float
        線形回帰の平均R²スコア。
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring='r2', n_jobs=-1)

    return float(np.mean(scores))


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
