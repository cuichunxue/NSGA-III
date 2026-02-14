"""
NSGA-III 多目的最適化 コアモジュール

joblib保存済み sm.OLS モデルを用いたパレート最適解の探索、
および pymoo 組み込みテスト問題のサポート。
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

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
    """目的関数値（目標値からの逸脱量）を計算する。"""

    def __init__(
        self,
        predictor: OLSModelPredictor,
        targets: Dict[str, float],
        aggregation_mode: str,
    ):
        self.predictor = predictor
        self.targets = targets
        self.aggregation_mode = aggregation_mode
        self.target_names = predictor.target_names

    def compute_objectives(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Y, _ = self.predictor.predict(X)
        deviations = np.zeros_like(Y)
        for i, name in enumerate(self.target_names):
            target = self.targets.get(name, 0.0)
            deviations[:, i] = np.abs(Y[:, i] - target)

        if self.aggregation_mode == "full":
            return deviations, Y
        else:
            mean_dev = np.mean(deviations, axis=1, keepdims=True)
            max_dev = np.max(deviations, axis=1, keepdims=True)
            return np.hstack([mean_dev, max_dev]), Y

    def get_n_objectives(self) -> int:
        if self.aggregation_mode == "full":
            return len(self.target_names)
        return 2

    def get_objective_names(self) -> List[str]:
        if self.aggregation_mode == "full":
            return [f"dev_{name}" for name in self.target_names]
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
