"""
NSGA-III 多目的最適化 コアモジュール

joblib保存済み sm.OLS モデルを用いたパレート最適解の探索、
および pymoo 組み込みテスト問題のサポート。
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
from pymoo.core.problem import Problem


class FeatureCalculator:
    """OLS モデル用の特徴量を計算する。"""

    def __init__(self, variable_names: List[str], center_values: Dict[str, float]):
        self.variable_names = variable_names
        self.center_values = center_values
        self.var_to_idx = {name: i for i, name in enumerate(variable_names)}

    def compute_features(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        n_samples = X.shape[0]
        features = np.zeros((n_samples, len(feature_names)))
        for i, feat_name in enumerate(feature_names):
            features[:, i] = self._compute_single_feature(X, feat_name)
        return features

    def _compute_single_feature(self, X: np.ndarray, feat_name: str) -> np.ndarray:
        n_samples = X.shape[0]

        # 定数項
        if feat_name == "const":
            return np.ones(n_samples)

        # 二乗項: "VAR^2 (centered)"
        if "^2" in feat_name and "(centered)" in feat_name:
            var_name = feat_name.split("^2")[0]
            if var_name in self.var_to_idx:
                idx = self.var_to_idx[var_name]
                c = self.center_values.get(var_name, 0.0)
                return (X[:, idx] - c) ** 2
            else:
                warnings.warn(f"変数が見つかりません: {var_name}")
                return np.zeros(n_samples)

        # 交互作用項: "VAR1:VAR2 (centered)"
        if ":" in feat_name and "(centered)" in feat_name:
            vars_part = feat_name.replace(" (centered)", "")
            var_names = vars_part.split(":")
            if len(var_names) == 2:
                var1, var2 = var_names
                if var1 in self.var_to_idx and var2 in self.var_to_idx:
                    idx1 = self.var_to_idx[var1]
                    idx2 = self.var_to_idx[var2]
                    c1 = self.center_values.get(var1, 0.0)
                    c2 = self.center_values.get(var2, 0.0)
                    return (X[:, idx1] - c1) * (X[:, idx2] - c2)
                else:
                    warnings.warn(f"変数が見つかりません: {var1} or {var2}")
                    return np.zeros(n_samples)

        # 一次項
        if feat_name in self.var_to_idx:
            idx = self.var_to_idx[feat_name]
            return X[:, idx]

        warnings.warn(f"未知の特徴量パターン: {feat_name}")
        return np.zeros(n_samples)


class OLSModelPredictor:
    """joblib 保存済み OLS モデルで予測する。"""

    def __init__(
        self,
        model_paths: Dict[str, str],
        variable_names: List[str],
        center_values: Dict[str, float],
    ):
        self.variable_names = variable_names
        self.feature_calculator = FeatureCalculator(variable_names, center_values)
        self.target_names: List[str] = []
        self.models: Dict[str, object] = {}
        self.model_features: Dict[str, List[str]] = {}
        self.model_params: Dict[str, np.ndarray] = {}
        self._load_models(model_paths)

    def _load_models(self, model_paths: Dict[str, str]):
        for target_name, path in model_paths.items():
            p = Path(path)
            if not p.exists():
                warnings.warn(f"ファイルが見つかりません: {path}")
                continue
            model = joblib.load(p)
            self.models[target_name] = model
            self.target_names.append(target_name)

            if hasattr(model, "params") and hasattr(model.params, "index"):
                feature_names = model.params.index.tolist()
                self.model_features[target_name] = feature_names
                self.model_params[target_name] = model.params.values
            else:
                warnings.warn(f"特徴量名を取得できません: {target_name}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        n_samples = X.shape[0]
        n_outputs = len(self.target_names)
        Y = np.zeros((n_samples, n_outputs))

        for i, target_name in enumerate(self.target_names):
            if target_name in self.model_features:
                feature_names = self.model_features[target_name]
                params = self.model_params[target_name]
                features = self.feature_calculator.compute_features(X, feature_names)
                Y[:, i] = features @ params
            else:
                Y[:, i] = np.nan

        return Y, self.target_names


class ObjectiveCalculator:
    """目的関数値を計算する。

    各目的変数に対して以下の方向をサポート:
      - "target": 目標値からの逸脱量 |Y - target| を最小化（既定）
      - "maximize": 予測値 Y を最大化（内部的には -Y を最小化）
      - "minimize": 予測値 Y を最小化
    """

    def __init__(
        self,
        predictor: OLSModelPredictor,
        targets: Dict[str, float],
        aggregation_mode: str,
        deviation_mode: str = "absolute",
        directions: Optional[Dict[str, str]] = None,
    ):
        self.predictor = predictor
        self.targets = targets
        self.deviation_mode = deviation_mode
        self.target_names = predictor.target_names
        self.directions = directions or {}

        # maximize/minimize が含まれる場合は aggregated モードが意味をなさないため
        # 自動的に full モードに切り替える
        has_non_target = any(
            self.directions.get(name, "target") != "target"
            for name in self.target_names
        )
        if has_non_target and aggregation_mode == "aggregated":
            self.aggregation_mode = "full"
        else:
            self.aggregation_mode = aggregation_mode

    def compute_objectives(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Y, _ = self.predictor.predict(X)
        objectives = np.zeros_like(Y)

        for i, name in enumerate(self.target_names):
            direction = self.directions.get(name, "target")

            if direction == "maximize":
                objectives[:, i] = -Y[:, i]
            elif direction == "minimize":
                objectives[:, i] = Y[:, i]
            else:  # "target"
                target = self.targets.get(name, 0.0)
                abs_dev = np.abs(Y[:, i] - target)
                if self.deviation_mode == "normalized" and abs(target) > 1e-12:
                    objectives[:, i] = abs_dev / abs(target)
                else:
                    objectives[:, i] = abs_dev

        if self.aggregation_mode == "full":
            return objectives, Y
        else:
            mean_obj = np.mean(objectives, axis=1, keepdims=True)
            max_obj = np.max(objectives, axis=1, keepdims=True)
            return np.hstack([mean_obj, max_obj]), Y

    def get_n_objectives(self) -> int:
        if self.aggregation_mode == "full":
            return len(self.target_names)
        return 2

    def get_objective_names(self) -> List[str]:
        if self.aggregation_mode == "full":
            names = []
            for name in self.target_names:
                direction = self.directions.get(name, "target")
                if direction == "maximize":
                    names.append(f"neg_{name}")
                elif direction == "minimize":
                    names.append(name)
                else:
                    names.append(f"dev_{name}")
            return names
        return ["mean_dev", "max_dev"]


class MultiObjectiveProblem(Problem):
    """pymoo 用の多目的最適化問題定義。"""

    def __init__(
        self,
        obj_calculator: ObjectiveCalculator,
        xl: np.ndarray,
        xu: np.ndarray,
    ):
        self.obj_calculator = obj_calculator
        super().__init__(
            n_var=len(xl),
            n_obj=obj_calculator.get_n_objectives(),
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            vtype=float,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F, _ = self.obj_calculator.compute_objectives(X)
        out["F"] = F


class RankingObjectiveCalculator:
    """ランキング一致を目的関数として計算する。

    予測ランキングと実機ランキングの Kendall 距離および
    連続マージン補正項を最小化する単一スカラー目的関数を提供する。

    同点（タイ）処理:
        np.argsort に stable=True を指定し、同値はグループ内インデックス昇順で処理する。
        すなわち、値が等しい場合は元のグループインデックスが小さい要素が上位とみなされる。
    """

    def __init__(
        self,
        predictor: OLSModelPredictor,
        groups: List[List[str]],
        real_rankings: List[List[int]],
        weights: List[float],
        margin_lambda: float = 0.01,
        margin_delta: float = 0.01,
    ):
        """
        Parameters
        ----------
        predictor : OLSModelPredictor
        groups : List[List[str]]
            各グループ内の特性名リスト
        real_rankings : List[List[int]]
            各グループ内の実機順位（1-indexed, 1 が最良）
            real_rankings[g][i] = groups[g][i] の実機順位
        weights : List[float]
            グループ重み w_g
        margin_lambda : float
            連続マージン補正項の重み λ
        margin_delta : float
            マージン幅 δ
        """
        self.predictor = predictor
        self.groups = groups
        self.real_rankings = real_rankings
        self.weights = weights
        self.margin_lambda = margin_lambda
        self.margin_delta = margin_delta

        all_target_names = predictor.target_names

        # 各グループの特性インデックス（predictor.target_names 内）
        self._group_indices: List[List[int]] = []
        for group_members in groups:
            indices = []
            for name in group_members:
                if name in all_target_names:
                    indices.append(all_target_names.index(name))
                else:
                    raise ValueError(f"特性名がモデルに存在しません: {name}")
            self._group_indices.append(indices)

    @staticmethod
    def kendall_distance(ordering_a: List[int], ordering_b: List[int]) -> int:
        """2つの順序付けリストの Kendall 距離（逆転ペア数）を計算する。

        Parameters
        ----------
        ordering_a, ordering_b : 要素 0..n-1 の順列（先頭が最高位）

        Returns
        -------
        int : 逆転ペア数（0 以上 n*(n-1)/2 以下）
        """
        n = len(ordering_a)
        pos_b = {val: idx for idx, val in enumerate(ordering_b)}
        distance = 0
        for i in range(n):
            for j in range(i + 1, n):
                # ordering_a で ordering_a[i] が ordering_a[j] より前にあるのに
                # ordering_b では後ろにある場合を逆転とみなす
                if pos_b[ordering_a[i]] > pos_b[ordering_a[j]]:
                    distance += 1
        return distance

    def compute_objective(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """目的関数値 J(x)、全予測値 Y、グループ詳細を計算する。

        J(x) = Σ_g w_g * d_g(x) + λ * Σ_g C_g(x)

        Parameters
        ----------
        X : (n_samples, n_var)

        Returns
        -------
        J : (n_samples, 1) 目的関数値
        Y : (n_samples, K) 全特性の予測値
        group_details : List[dict] 最初のサンプルのグループ詳細（結果表示用）
        """
        Y, _ = self.predictor.predict(X)  # (n_samples, K)
        n_samples = X.shape[0]
        J = np.zeros(n_samples)
        delta = self.margin_delta

        group_details = []

        for g, (group_indices, weight) in enumerate(
            zip(self._group_indices, self.weights)
        ):
            ranks = self.real_rankings[g]
            ng = len(group_indices)
            group_Y = Y[:, group_indices]  # (n_samples, ng)

            d_g = np.zeros(n_samples)
            C_g = np.zeros(n_samples)

            # 全ペア (i, j) を走査 (i < j, グループ内インデックス)
            for i in range(ng):
                for j in range(i + 1, ng):
                    # 実機でどちらが上位か (ranks が小さいほど上位)
                    if ranks[i] < ranks[j]:
                        better, worse = i, j
                    else:
                        better, worse = j, i

                    # 予測値差: 上位予測 - 下位予測（正なら concordant 方向）
                    diff = group_Y[:, better] - group_Y[:, worse]  # (n_samples,)

                    # Kendall 距離: タイ処理（stable argsort 規則）
                    # stable 降順ソートでは同値の場合、元インデックス小さい方が先頭
                    # → better < worse のとき diff==0 は concordant
                    # → better > worse のとき diff==0 は discordant
                    if better < worse:
                        discordant_mask = diff < 0
                    else:
                        discordant_mask = diff <= 0

                    d_g += discordant_mask.astype(np.float64)

                    # 連続マージン補正項
                    concordant_mask = ~discordant_mask
                    # concordant pair: max(0, δ - diff)
                    C_g += np.where(
                        concordant_mask,
                        np.maximum(0.0, delta - diff),
                        0.0,
                    )
                    # discordant pair: -diff + δ（diff≦0 なので -diff≧0）
                    C_g += np.where(
                        discordant_mask,
                        -diff + delta,
                        0.0,
                    )

            J += weight * d_g + self.margin_lambda * C_g

            # 最初のサンプルのグループ詳細を記録（結果表示用）
            y0 = group_Y[0]
            pred_ordering = list(np.argsort(-y0, kind="stable"))
            real_ordering = sorted(range(ng), key=lambda k: ranks[k])

            group_details.append({
                "group_index": g,
                "members": self.groups[g],
                "real_ordering": real_ordering,
                "pred_ordering": pred_ordering,
                "kendall_distance": int(d_g[0]),
                "C_g": float(C_g[0]),
                "predicted_values": {
                    self.groups[g][k]: float(y0[k]) for k in range(ng)
                },
            })

        return J.reshape(-1, 1), Y, group_details

    def compute_group_distances(self, X: np.ndarray) -> np.ndarray:
        """各サンプル・各グループの Kendall 距離を一括計算する。

        Returns
        -------
        D : (n_samples, n_groups) int 配列
        """
        Y, _ = self.predictor.predict(X)
        n_samples = X.shape[0]
        n_groups = len(self.groups)
        D = np.zeros((n_samples, n_groups), dtype=int)

        for g, group_indices in enumerate(self._group_indices):
            ranks = self.real_rankings[g]
            ng = len(group_indices)
            group_Y = Y[:, group_indices]

            for i in range(ng):
                for j in range(i + 1, ng):
                    if ranks[i] < ranks[j]:
                        better, worse = i, j
                    else:
                        better, worse = j, i

                    diff = group_Y[:, better] - group_Y[:, worse]

                    if better < worse:
                        D[:, g] += (diff < 0).astype(int)
                    else:
                        D[:, g] += (diff <= 0).astype(int)

        return D


class RankingProblem(Problem):
    """pymoo 用のランキング最適化問題定義（単一目的）。"""

    def __init__(
        self,
        obj_calculator: RankingObjectiveCalculator,
        xl: np.ndarray,
        xu: np.ndarray,
    ):
        self.obj_calculator = obj_calculator
        super().__init__(
            n_var=len(xl),
            n_obj=1,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            vtype=float,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        J, _, _ = self.obj_calculator.compute_objective(X)
        out["F"] = J  # (n_samples, 1)
