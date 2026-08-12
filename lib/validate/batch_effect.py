from pathlib import Path
from typing import Dict, Tuple, Union

import h5py
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)
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


def compute_knn_accuracy_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_neighbors: int = 15,
    n_splits: int = 5,
    random_state: int = 42
    ) -> float:
    """
    StratifiedGroupKFoldを使用して、潜在表現のKNN分類器の精度を計算する関数。
    compute_knn_accuracyのグループ考慮版。

    compute_knn_accuracyはサンプル単位で無作為にfoldへ分割するため、例えば
    「同じスライド由来の複数パッチ」のように1グループが複数サンプルにまたがる
    データでは、同じスライドのパッチが訓練foldとテストfoldの両方に混入し、
    「そのスライド固有の見た目」を覚えるだけで精度が水増しされるリーク
    (leakage)が起きる。この関数はgroups（例: スライドID）でグループ化した上で
    fold分割することで、同じグループのサンプルが訓練/テストに分かれて入らない
    ようにする。

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。
    y : np.ndarray
        分類したいラベル（例: 病理所見の有無）。
    groups : np.ndarray
        サンプルが属するグループ（例: スライドID）。同じグループのサンプルは
        必ず同じfold（訓練 or テスト）に入る。
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
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean', n_jobs=1)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(knn, X, y, groups=groups, cv=sgkf, scoring='accuracy', n_jobs=-1)

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


def compute_knn_regression_r2_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_neighbors: int = 15,
    n_splits: int = 5,
    random_state: int = 42
    ) -> float:
    """
    GroupKFoldを使用して、潜在表現のKNN回帰R²を計算する関数。
    compute_knn_regression_r2のグループ考慮版。

    compute_knn_regression_r2は無作為にfoldへ分割するため、例えば
    「同じスライド由来の複数パッチ」のように1グループが複数サンプルに
    またがるデータでは、同じスライドのパッチが訓練foldとテストfoldの両方に
    混入するリークが起きる。UNI特徴量は「どのスライドか」をほぼ完璧に
    識別できてしまう（compute_knn_accuracy参照）ため、その気になれば
    「テストパッチがどのスライドかを当てて、訓練foldにある同じスライドの
    平均値を答える」というショートカットで精度が水増しされうる。この関数は
    groups（例: スライドID）でグループ化した上でfold分割することで、
    同じグループのサンプルが訓練/テストに分かれて入らないようにする。

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。
    y : np.ndarray
        連続値の概念ラベル（例: ぼやけスコア）。
    groups : np.ndarray
        サンプルが属するグループ（例: スライドID）。同じグループのサンプルは
        必ず同じfold（訓練 or テスト）に入る。
    n_neighbors : int, optional
        KNN回帰器の近傍数（デフォルトは15）。
    n_splits : int, optional
        クロスバリデーションの分割数（デフォルトは5）。
    random_state : int, optional
        乱数シード（デフォルトは42。GroupKFoldのシャッフルに使う）。

    Returns
    -------
    float
        KNN回帰器の平均R²スコア。
    """
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric='euclidean', n_jobs=1)
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(knn, X, y, groups=groups, cv=gkf, scoring='r2', n_jobs=-1)

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


def compute_linear_regression_r2_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
    ) -> float:
    """
    GroupKFoldを使用して、連続値の概念(y)の線形回帰R²を計算する関数。
    compute_linear_regression_r2のグループ考慮版（compute_knn_regression_r2_grouped
    と同じ理由でパッチ単位のリークを防ぐ）。

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。
    y : np.ndarray
        連続値の概念ラベル（例: ぼやけスコア）。
    groups : np.ndarray
        サンプルが属するグループ（例: スライドID）。
    n_splits : int, optional
        クロスバリデーションの分割数（デフォルトは5）。
    random_state : int, optional
        乱数シード（デフォルトは42。GroupKFoldのシャッフルに使う）。

    Returns
    -------
    float
        線形回帰の平均R²スコア。
    """
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(LinearRegression(), X, y, groups=groups, cv=gkf, scoring='r2', n_jobs=-1)

    return float(np.mean(scores))


def compute_logreg_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
    ) -> Dict[str, float]:
    """
    L2正則化ロジスティック回帰による二値分類プローブ。

    サンプル数が次元数より少ない（例: パッチをスライド単位で平均プーリングした
    特徴量はサンプル数=スライド数と少なくなりがち）ような高次元・少サンプルの
    状況では、KNNは次元の呪いで機能しにくい（実測でも同条件のKNNはchance level
    だった）が、正則化された線形モデルは安定して線形の手がかりを検出できる。
    クラス不均衡を考慮してclass_weight="balanced"を使い、balanced_accuracyと
    ROC-AUCの両方を返す（不均衡データでは素のaccuracyは多数派クラス予測だけで
    高くなってしまい当てにならないため）。

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。
    y : np.ndarray
        二値ラベル（例: 病理所見の有無）。
    n_splits : int, optional
        クロスバリデーションの分割数（デフォルトは5）。
    random_state : int, optional
        乱数シード（デフォルトは42）。

    Returns
    -------
    Dict[str, float]
        "balanced_accuracy": 平均balanced accuracy。
        "roc_auc": 平均ROC-AUC。
    """
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    balanced_acc = cross_val_score(clf, X, y, cv=skf, scoring='balanced_accuracy', n_jobs=-1)
    roc_auc = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)

    return {
        "balanced_accuracy": float(np.mean(balanced_acc)),
        "roc_auc": float(np.mean(roc_auc)),
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



def compute_effective_rank(X: np.ndarray) -> Dict[str, float]:
    """
    表現Xの有効ランク(effective rank)を計算する関数（Roy & Vetterli, 2007）。

    共分散行列の固有値から求めた特異値分布のシャノンエントロピーの指数として
    定義される。通常の(整数の)ランクと違い、「支配的な方向が実質何本あるか」を
    連続値で捉える指標で、1本の方向に分散が集中していれば1に近づき、
    全次元に均等に分散が広がっていれば全次元数に近づく。

    erasureで特定の方向を潰すと、この値も潰れた分だけ下がる（＝表現が
    低ランク化した/情報の多様性が失われた）ことを検出できる。

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。

    Returns
    -------
    Dict[str, float]
        "effective_rank": 有効ランクの値そのもの（1〜Dの範囲）。
        "effective_rank_ratio": 全次元数Dで正規化した比率（0〜1の範囲）。
    """
    total_dim = X.shape[1]

    X_centered = X - X.mean(axis=0)
    cov = (X_centered.T @ X_centered) / (len(X_centered) - 1)
    # 共分散行列は対称なのでeigvalshを使う（数値誤差で出る微小な負値はクリップ）。
    eigenvalues = np.clip(np.linalg.eigvalsh(cov), a_min=0, a_max=None)

    # 特異値 = sqrt(固有値)。定数倍(sqrt(n-1))は正規化で相殺されるため省略できる。
    singular_values = np.sqrt(eigenvalues)
    singular_values = singular_values[singular_values > 1e-12]

    p = singular_values / singular_values.sum()
    entropy = -np.sum(p * np.log(p))
    effective_rank = float(np.exp(entropy))

    return {
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank / total_dim,
    }


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
