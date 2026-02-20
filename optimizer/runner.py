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
    RankingObjectiveCalculator,
    RankingProblem,
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


def predict_single(config: Dict[str, Any], values: Dict[str, float]) -> Dict[str, Any]:
    """単一の変数値セットに対して予測・目的関数値を計算する。"""
    mode = config.get("mode", "demo")

    if mode == "demo":
        problem_key = config.get("problem", "zdt1")
        if problem_key not in DEMO_PROBLEMS:
            raise ValueError(f"不明なデモ問題: {problem_key}")
        info = DEMO_PROBLEMS[problem_key]
        problem = _get_demo_problem(problem_key)

        var_names = [f"x{i+1}" for i in range(info["n_var"])]
        X = np.array([[values.get(v, 0.0) for v in var_names]])
        F = problem.evaluate(X)

        obj_names = info["obj_names"]
        result = {"variables": {}, "objectives": {}}
        for i, name in enumerate(var_names):
            result["variables"][name] = round(float(X[0, i]), 6)
        for i, name in enumerate(obj_names):
            result["objectives"][name] = round(float(F[0, i]), 6)
        return result

    if mode == "ranking":
        variables = config["variables"]
        model_paths = config["model_paths"]
        groups = config.get("groups", [])
        real_rankings = config.get("real_rankings", [])

        variable_names = list(variables.keys())
        center_values = {k: v[2] for k, v in variables.items()}

        predictor = OLSModelPredictor(model_paths, variable_names, center_values)
        X = np.array([[values.get(v, 0.0) for v in variable_names]])
        Y, target_names = predictor.predict(X)

        result: Dict[str, Any] = {
            "variables": {},
            "predictions": {},
            "ranking_results": [],
            "total_kendall": 0,
        }
        for i, name in enumerate(variable_names):
            result["variables"][name] = round(float(X[0, i]), 6)
        for i, name in enumerate(target_names):
            result["predictions"][name] = {"predicted": round(float(Y[0, i]), 6)}

        total_kendall = 0
        for g, (group_members, real_ranks) in enumerate(zip(groups, real_rankings)):
            ng = len(group_members)
            # モデルがロードされていない場合は 0.0 にフォールバック（len は常に ng）
            y_g = np.array([
                float(Y[0, target_names.index(name)]) if name in target_names else 0.0
                for name in group_members
            ])

            pred_ordering = list(np.argsort(-y_g, kind="stable"))
            real_ordering = sorted(range(ng), key=lambda k: real_ranks[k])

            # 予測ランキング（1-indexed、メンバーごと）
            pred_ranks = [0] * ng
            for rank_pos, member_idx in enumerate(pred_ordering):
                pred_ranks[member_idx] = rank_pos + 1

            d = RankingObjectiveCalculator.kendall_distance(pred_ordering, real_ordering)
            total_kendall += d

            # 逆転ペアを列挙
            discordant_pairs = []
            for i in range(ng):
                for j in range(i + 1, ng):
                    if real_ranks[i] < real_ranks[j]:
                        better, worse = i, j
                    else:
                        better, worse = j, i
                    # discordant: better の予測値 <= worse の予測値
                    if y_g[better] < y_g[worse] or (
                        y_g[better] == y_g[worse] and better > worse
                    ):
                        discordant_pairs.append(
                            [group_members[better], group_members[worse]]
                        )

            result["ranking_results"].append({
                "group_index": g,
                "members": group_members,
                "real_ranking": real_ranks,
                "pred_ranking": pred_ranks,
                "kendall_distance": d,
                "discordant_pairs": discordant_pairs,
            })

        result["total_kendall"] = total_kendall
        return result

    # custom mode
    variables = config["variables"]
    model_paths = config["model_paths"]
    targets = config["targets"]
    dev_mode = config.get("deviation_mode", "absolute")
    directions = config.get("directions", {})

    variable_names = list(variables.keys())
    center_values = {k: v[2] for k, v in variables.items()}

    predictor = OLSModelPredictor(model_paths, variable_names, center_values)
    X = np.array([[values.get(v, 0.0) for v in variable_names]])
    Y, target_names = predictor.predict(X)

    result = {"variables": {}, "predictions": {}, "deviations": {}, "objectives": {}}
    for i, name in enumerate(variable_names):
        result["variables"][name] = round(float(X[0, i]), 6)
    for i, name in enumerate(target_names):
        pred_val = float(Y[0, i])
        direction = directions.get(name, "target")
        target_val = targets.get(name, 0.0)

        entry = {
            "predicted": round(pred_val, 6),
            "direction": direction,
        }

        if direction == "target":
            abs_dev = abs(pred_val - target_val)
            if dev_mode == "normalized" and abs(target_val) > 1e-12:
                dev = abs_dev / abs(target_val)
            else:
                dev = abs_dev
            entry["target"] = target_val
            entry["abs_deviation"] = round(abs_dev, 6)
            entry["deviation"] = round(dev, 6)
        else:
            entry["target"] = None
            entry["abs_deviation"] = None
            entry["deviation"] = None

        result["predictions"][name] = entry
    return result


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
        elif mode == "ranking":
            return self._run_ranking()
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
        variables = self.config.get("variables")
        model_paths = self.config.get("model_paths")
        targets = self.config.get("targets")

        if not variables:
            raise RuntimeError("設計変数が定義されていません。")
        if not model_paths:
            raise RuntimeError("回帰モデルが指定されていません。")
        if not targets:
            raise RuntimeError("目標値が定義されていません。")

        # モデルと目標値の整合性を検証
        model_keys = set(model_paths.keys())
        target_keys = set(targets.keys())
        models_without_targets = model_keys - target_keys
        if models_without_targets:
            raise RuntimeError(
                f"以下のモデルに対応する目標値が未定義です: "
                f"{', '.join(sorted(models_without_targets))}"
            )
        targets_without_models = target_keys - model_keys
        if targets_without_models:
            raise RuntimeError(
                f"以下の目標値に対応するモデルがありません: "
                f"{', '.join(sorted(targets_without_models))}"
            )

        pop_size = int(self.config.get("pop_size", 200))
        n_gen = int(self.config.get("n_gen", 300))
        seed = int(self.config.get("seed", 42))
        agg_mode = self.config.get("aggregation_mode", "aggregated")
        dev_mode = self.config.get("deviation_mode", "absolute")
        directions = self.config.get("directions", {})

        variable_names = list(variables.keys())
        xl = np.array([v[0] for v in variables.values()])
        xu = np.array([v[1] for v in variables.values()])
        center_values = {k: v[2] for k, v in variables.items()}

        predictor = OLSModelPredictor(model_paths, variable_names, center_values)
        if not predictor.target_names:
            raise RuntimeError("有効なモデルがロードされませんでした。ファイルパスを確認してください。")

        obj_calculator = ObjectiveCalculator(
            predictor, targets, agg_mode, dev_mode, directions,
        )
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
            directions=directions,
        )

    # ---- ranking (single-objective GA) mode ------------------------------

    def _run_ranking(self) -> Dict[str, Any]:
        """ランキング最適化を実行する（単一目的 GA）。"""
        from pymoo.algorithms.soo.nonconvex.ga import GA

        config = self.config
        variables = config.get("variables")
        model_paths = config.get("model_paths")
        groups = config.get("groups")
        real_rankings = config.get("real_rankings")

        if not variables:
            raise RuntimeError("設計変数が定義されていません。")
        if not model_paths:
            raise RuntimeError("回帰モデルが指定されていません。")
        if not groups:
            raise RuntimeError("グループが定義されていません。")
        if not real_rankings:
            raise RuntimeError("実機ランキングが定義されていません。")

        pop_size = int(config.get("pop_size", 300))
        n_gen = int(config.get("n_gen", 500))
        seed = int(config.get("seed", 42))
        group_weights = config.get("group_weights") or [1.0] * len(groups)
        margin_lambda = float(config.get("margin_lambda", 0.01))
        margin_delta = float(config.get("margin_delta", 0.01))

        variable_names = list(variables.keys())
        xl = np.array([v[0] for v in variables.values()])
        xu = np.array([v[1] for v in variables.values()])
        center_values = {k: v[2] for k, v in variables.items()}

        predictor = OLSModelPredictor(model_paths, variable_names, center_values)
        if not predictor.target_names:
            raise RuntimeError("有効なモデルがロードされませんでした。ファイルパスを確認してください。")

        obj_calculator = RankingObjectiveCalculator(
            predictor, groups, real_rankings, group_weights,
            margin_lambda, margin_delta,
        )
        problem = RankingProblem(obj_calculator, xl, xu)

        # 収束履歴を記録する
        convergence_history: List[float] = []

        def on_progress_with_history(gen, total, best_val):
            if best_val is not None:
                convergence_history.append(round(float(best_val), 6))
            self.progress_callback(gen, total, best_val)

        algorithm = GA(pop_size=pop_size)
        termination = get_termination("n_gen", n_gen)
        callback = _ProgressCallback(n_gen, on_progress_with_history)

        result = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False,
            save_history=False,
            callback=callback,
        )

        # 最終世代の全個体から J 値昇順で上位 20 個を抽出
        final_X = result.pop.get("X")
        final_F = result.pop.get("F")
        sort_idx = np.argsort(final_F[:, 0])[:20]
        top_X = final_X[sort_idx]
        top_F = final_F[sort_idx]

        return self._build_ranking_result(
            top_X, top_F, variable_names,
            obj_calculator, groups, real_rankings, convergence_history,
        )

    def _build_ranking_result(
        self,
        top_X: np.ndarray,
        top_F: np.ndarray,
        variable_names: List[str],
        obj_calculator: "RankingObjectiveCalculator",
        groups: List[List[str]],
        real_rankings: List[List[int]],
        convergence_history: List[float],
    ) -> Dict[str, Any]:
        """ランキング最適化の結果を JSON シリアライズ可能な辞書に整形する。"""
        n_top = len(top_X)

        # 最良解（最小 J の解）のグループ詳細を計算
        _, _, group_details = obj_calculator.compute_objective(top_X[:1])

        # グループ結果を構築
        group_results = []
        for g, detail in enumerate(group_details):
            ng = len(groups[g])
            max_kendall = ng * (ng - 1) // 2
            d_g = detail["kendall_distance"]

            # pred_ranking（1-indexed、メンバーごと）
            pred_ordering = detail["pred_ordering"]
            pred_ranks = [0] * ng
            for rank_pos, member_idx in enumerate(pred_ordering):
                pred_ranks[member_idx] = rank_pos + 1

            if d_g == 0:
                status = "perfect"
            elif max_kendall > 0 and d_g >= max_kendall / 2:
                status = "mismatch"
            else:
                status = "partial"

            group_results.append({
                "group_index": g,
                "members": groups[g],
                "real_ranking": real_rankings[g],
                "pred_ranking": pred_ranks,
                "kendall_distance": d_g,
                "max_kendall_distance": max_kendall,
                "predicted_values": detail["predicted_values"],
                "status": status,
            })

        total_kendall = sum(gr["kendall_distance"] for gr in group_results)
        max_kendall_total = sum(gr["max_kendall_distance"] for gr in group_results)

        # 上位解ごとのグループ Kendall 距離を一括計算
        D_all = obj_calculator.compute_group_distances(top_X)  # (n_top, n_groups)

        table_rows = []
        for row_idx in range(n_top):
            row: Dict[str, Any] = {}
            for i, name in enumerate(variable_names):
                row[name] = round(float(top_X[row_idx, i]), 6)
            row["J_rank"] = round(float(top_F[row_idx, 0]), 6)
            for g in range(len(groups)):
                row[f"d_G{g + 1}"] = int(D_all[row_idx, g])
            table_rows.append(row)

        column_order = list(variable_names) + ["J_rank"]
        column_order += [f"d_G{g + 1}" for g in range(len(groups))]

        perfect = sum(1 for gr in group_results if gr["status"] == "perfect")
        partial = sum(1 for gr in group_results if gr["status"] == "partial")
        mismatch = sum(1 for gr in group_results if gr["status"] == "mismatch")

        return {
            "mode": "ranking",
            "n_solutions": n_top,
            "var_names": variable_names,
            "best_J": round(float(top_F[0, 0]), 6),
            "total_kendall": total_kendall,
            "max_kendall": max_kendall_total,
            "group_results": group_results,
            "table": table_rows,
            "column_order": column_order,
            "convergence_history": convergence_history,
            "summary": {
                "n_solutions": n_top,
                "perfect_groups": perfect,
                "partial_groups": partial,
                "mismatch_groups": mismatch,
            },
        }

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
        directions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """結果を JSON シリアライズ可能な辞書に整形。"""
        directions = directions or {}
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
                direction = directions.get(name, "target")
                if direction == "target":
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
            "directions": directions,
            "summary": summary,
            "table": table_rows,
        }
