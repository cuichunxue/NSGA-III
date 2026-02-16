"""
最適化実行エンジン

デモ問題（pymoo 組み込み）とカスタム OLS モデル問題の両方をサポート。
進捗コールバック付き。
"""

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from .core import (
    MultiObjectiveProblem,
    OLSModelPredictor,
    ObjectiveCalculator,
)

# ---------------------------------------------------------------------------
# デモ問題の定義
# ---------------------------------------------------------------------------

DEMO_PROBLEMS = {
    "zdt1": {
        "name": "ZDT1",
        "description": "凸型パレートフロント（2目的・30変数）。多目的最適化の基本ベンチマーク。",
        "n_var": 30,
        "n_obj": 2,
        "obj_names": ["f1", "f2"],
    },
    "zdt2": {
        "name": "ZDT2",
        "description": "非凸型パレートフロント（2目的・30変数）。凹形状のフロントを探索。",
        "n_var": 30,
        "n_obj": 2,
        "obj_names": ["f1", "f2"],
    },
    "zdt3": {
        "name": "ZDT3",
        "description": "不連続パレートフロント（2目的・30変数）。分断されたフロントの探索。",
        "n_var": 30,
        "n_obj": 2,
        "obj_names": ["f1", "f2"],
    },
    "dtlz2": {
        "name": "DTLZ2",
        "description": "球面パレートフロント（3目的・10変数）。3D可視化が可能。",
        "n_var": 10,
        "n_obj": 3,
        "obj_names": ["f1", "f2", "f3"],
    },
    "kursawe": {
        "name": "Kursawe",
        "description": "非凸・不連続フロント（2目的・3変数）。変数が少なく高速。",
        "n_var": 3,
        "n_obj": 2,
        "obj_names": ["f1", "f2"],
    },
}


def _get_demo_problem(name: str):
    """pymoo の組み込みテスト問題を返す。"""
    from pymoo.problems import get_problem

    if name == "kursawe":
        return get_problem("kursawe")
    return get_problem(name)


# ---------------------------------------------------------------------------
# 進捗コールバック
# ---------------------------------------------------------------------------

class _ProgressCallback(Callback):
    """各世代で進捗情報を通知するコールバック。"""

    def __init__(self, n_gen_total: int, notify_fn: Callable):
        super().__init__()
        self._n_gen_total = n_gen_total
        self._notify_fn = notify_fn

    def notify(self, algorithm):
        gen = algorithm.n_gen
        # 現世代の最良目的関数値
        pop_f = algorithm.pop.get("F")
        best_f = float(np.min(np.sum(pop_f, axis=1))) if pop_f is not None else None
        self._notify_fn(gen, self._n_gen_total, best_f)


# ---------------------------------------------------------------------------
# メインランナー
# ---------------------------------------------------------------------------

class OptimizationRunner:
    """NSGA-III 最適化を実行し、結果を構造化して返す。"""

    def __init__(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ):
        self.config = config
        self.progress_callback = progress_callback or (lambda *a: None)

    # ---- public -----------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """最適化を実行し結果辞書を返す。"""
        mode = self.config.get("mode", "demo")
        if mode == "demo":
            return self._run_demo()
        elif mode == "custom":
            return self._run_custom()
        else:
            raise ValueError(f"不明なモード: {mode}")

    # ---- demo mode --------------------------------------------------------

    def _run_demo(self) -> Dict[str, Any]:
        problem_key = self.config.get("problem", "zdt1")
        if problem_key not in DEMO_PROBLEMS:
            raise ValueError(f"不明なデモ問題: {problem_key}（選択可能: {', '.join(DEMO_PROBLEMS)}）")

        pop_size = int(self.config.get("pop_size", 100))
        n_gen = int(self.config.get("n_gen", 200))
        seed = int(self.config.get("seed", 42))

        info = DEMO_PROBLEMS[problem_key]
        problem = _get_demo_problem(problem_key)
        n_obj = info["n_obj"]

        ref_dirs = self._make_ref_dirs(n_obj)
        effective_pop = max(pop_size, len(ref_dirs))

        algorithm = NSGA3(pop_size=effective_pop, ref_dirs=ref_dirs)
        termination = get_termination("n_gen", n_gen)

        callback = _ProgressCallback(n_gen, self.progress_callback)

        result = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False,
            save_history=False,
            callback=callback,
        )

        pareto_X = result.X
        pareto_F = result.F

        var_names = [f"x{i+1}" for i in range(pareto_X.shape[1])]
        obj_names = info["obj_names"]

        return self._build_result(
            pareto_X, pareto_F, var_names, obj_names,
            predictions=None, target_names=None, targets=None,
        )

    # ---- custom (OLS) mode ------------------------------------------------

    def _run_custom(self) -> Dict[str, Any]:
        variables = self.config["variables"]  # {name: (lo, hi, center)}
        model_paths = self.config["model_paths"]  # {target: path}
        targets = self.config["targets"]  # {target: value}
        pop_size = int(self.config.get("pop_size", 200))
        n_gen = int(self.config.get("n_gen", 300))
        seed = int(self.config.get("seed", 42))
        agg_mode = self.config.get("aggregation_mode", "aggregated")
        dev_mode = self.config.get("deviation_mode", "absolute")

        variable_names = list(variables.keys())
        xl = np.array([v[0] for v in variables.values()])
        xu = np.array([v[1] for v in variables.values()])
        center_values = {k: v[2] for k, v in variables.items()}

        predictor = OLSModelPredictor(model_paths, variable_names, center_values)
        if not predictor.target_names:
            raise RuntimeError("有効なモデルがロードされませんでした。ファイルパスを確認してください。")

        obj_calculator = ObjectiveCalculator(predictor, targets, agg_mode, dev_mode)
        problem = MultiObjectiveProblem(obj_calculator, xl, xu)
        n_obj = obj_calculator.get_n_objectives()

        ref_dirs = self._make_ref_dirs(n_obj)
        effective_pop = max(pop_size, len(ref_dirs))

        algorithm = NSGA3(pop_size=effective_pop, ref_dirs=ref_dirs)
        termination = get_termination("n_gen", n_gen)
        callback = _ProgressCallback(n_gen, self.progress_callback)

        result = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False,
            save_history=False,
            callback=callback,
        )

        pareto_X = result.X
        pareto_F = result.F
        pareto_Y, target_names = predictor.predict(pareto_X)

        obj_names = obj_calculator.get_objective_names()

        return self._build_result(
            pareto_X, pareto_F, variable_names, obj_names,
            predictions=pareto_Y, target_names=target_names, targets=targets,
        )

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _make_ref_dirs(n_obj: int) -> np.ndarray:
        if n_obj <= 2:
            n_partitions = 12
        elif n_obj <= 3:
            n_partitions = 12
        elif n_obj <= 5:
            n_partitions = 6
        elif n_obj <= 10:
            n_partitions = 3
        else:
            n_partitions = 2
        return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)

    @staticmethod
    def _build_result(
        pareto_X: np.ndarray,
        pareto_F: np.ndarray,
        var_names: List[str],
        obj_names: List[str],
        predictions: Optional[np.ndarray] = None,
        target_names: Optional[List[str]] = None,
        targets: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """結果を JSON シリアライズ可能な辞書に整形。"""
        n_solutions = len(pareto_X)

        # 変数値
        variables_data = {}
        for i, name in enumerate(var_names):
            variables_data[name] = pareto_X[:, i].tolist()

        # 目的関数値
        objectives_data = {}
        for i, name in enumerate(obj_names):
            objectives_data[name] = pareto_F[:, i].tolist()

        # 予測値（カスタムモードのみ）
        predictions_data = {}
        deviations_data = {}
        if predictions is not None and target_names is not None and targets is not None:
            for i, name in enumerate(target_names):
                predictions_data[f"pred_{name}"] = predictions[:, i].tolist()
                target_val = targets.get(name, 0.0)
                deviations_data[f"dev_{name}"] = np.abs(
                    predictions[:, i] - target_val
                ).tolist()

        # サマリー統計
        summary = {
            "n_solutions": n_solutions,
            "variables": {},
            "objectives": {},
        }
        for i, name in enumerate(var_names):
            col = pareto_X[:, i]
            summary["variables"][name] = {
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "mean": float(np.mean(col)),
            }
        for i, name in enumerate(obj_names):
            col = pareto_F[:, i]
            summary["objectives"][name] = {
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "mean": float(np.mean(col)),
            }

        # CSV 用テーブルデータ（行指向）
        table_rows = []
        for row_idx in range(n_solutions):
            row = {}
            for i, name in enumerate(var_names):
                row[name] = round(float(pareto_X[row_idx, i]), 6)
            if predictions is not None and target_names is not None:
                for i, name in enumerate(target_names):
                    row[f"pred_{name}"] = round(float(predictions[row_idx, i]), 6)
            for i, name in enumerate(obj_names):
                row[name] = round(float(pareto_F[row_idx, i]), 6)
            table_rows.append(row)

        # テーブル列順序（フロントエンド用）
        column_order = list(var_names)
        if predictions is not None and target_names is not None:
            column_order += [f"pred_{n}" for n in target_names]
        column_order += list(obj_names)

        return {
            "n_solutions": n_solutions,
            "var_names": var_names,
            "obj_names": obj_names,
            "column_order": column_order,
            "variables": variables_data,
            "objectives": objectives_data,
            "predictions": predictions_data,
            "deviations": deviations_data,
            "summary": summary,
            "table": table_rows,
        }
