#!/usr/bin/env python3
"""
NSGA-III 多目的最適化 Web アプリケーション

Flask ベースの初心者向け最適化ツール。
デモ問題での即時体験と、カスタム OLS モデルによる本格運用の両方をサポート。
"""

import copy
import csv
import io
import os
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
)

from optimizer.runner import DEMO_PROBLEMS, OptimizationRunner, predict_single

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
app.json.sort_keys = False  # テーブル列の挿入順序を維持

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 古いジョブを自動削除するまでの秒数
_JOB_TTL = 3600  # 1 時間

# ---------------------------------------------------------------------------
# インメモリジョブストア（シングルプロセス用）
# ---------------------------------------------------------------------------

_jobs: dict = {}
_lock = threading.Lock()


def _update_job(job_id: str, **kwargs):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _get_job_snapshot(job_id: str):
    """ロック内でジョブのスナップショットを取得する（スレッドセーフ）。"""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return copy.copy(job)


def _cleanup_old_jobs():
    """TTL 超過のジョブを削除する。"""
    now = time.time()
    with _lock:
        expired = [
            jid for jid, j in _jobs.items()
            if now - j["started_at"] > _JOB_TTL and j["status"] != "running"
        ]
        for jid in expired:
            del _jobs[jid]


# ---------------------------------------------------------------------------
# ルーティング
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/demo-problems")
def api_demo_problems():
    """利用可能なデモ問題の一覧を返す。"""
    return jsonify(DEMO_PROBLEMS)


# ---- 最適化の開始 ---------------------------------------------------------

@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    """最適化ジョブを開始する。"""
    try:
        config = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "無効なJSONリクエストです"}), 400

    if not isinstance(config, dict):
        return jsonify({"error": "リクエストはJSON辞書である必要があります"}), 400

    # カスタムモードの厳密バリデーション
    mode = config.get("mode", "demo")
    if mode == "custom":
        err = _validate_custom_config(config)
        if err:
            return jsonify({"error": err}), 400

    n_gen = int(config.get("n_gen", 200))

    # 古いジョブを定期的に掃除
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex[:10]

    with _lock:
        _jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "current_gen": 0,
            "total_gen": n_gen,
            "best_value": None,
            "result": None,
            "error": None,
            "config": config,
            "started_at": time.time(),
        }

    t = threading.Thread(target=_run_job, args=(job_id, config), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


def _run_job(job_id: str, config: dict):
    """バックグラウンドスレッドで最適化を実行。"""

    def on_progress(gen, total, best_val):
        pct = int(gen / total * 100) if total > 0 else 0
        _update_job(
            job_id,
            progress=pct,
            current_gen=gen,
            total_gen=total,
            best_value=best_val,
        )

    try:
        runner = OptimizationRunner(config, progress_callback=on_progress)
        result = runner.run()
        _update_job(job_id, status="completed", progress=100, result=result)
    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc))


# ---- 進捗確認 -------------------------------------------------------------

@app.route("/api/progress/<job_id>")
def api_progress(job_id: str):
    job = _get_job_snapshot(job_id)
    if job is None:
        return jsonify({"error": "ジョブが見つかりません"}), 404

    elapsed = time.time() - job["started_at"]
    return jsonify({
        "status": job["status"],
        "progress": job["progress"],
        "current_gen": job["current_gen"],
        "total_gen": job["total_gen"],
        "best_value": job["best_value"],
        "elapsed": round(elapsed, 1),
        "error": job["error"],
    })


# ---- 結果取得 --------------------------------------------------------------

@app.route("/api/results/<job_id>")
def api_results(job_id: str):
    job = _get_job_snapshot(job_id)
    if job is None:
        return jsonify({"error": "ジョブが見つかりません"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "最適化がまだ完了していません"}), 400
    return jsonify(job["result"])


# ---- CSV ダウンロード -------------------------------------------------------

@app.route("/api/download/<job_id>")
def api_download(job_id: str):
    job = _get_job_snapshot(job_id)
    if job is None or job["status"] != "completed":
        return jsonify({"error": "結果がありません"}), 404

    result = job["result"]
    table = result.get("table", [])
    if not table:
        return jsonify({"error": "データがありません"}), 404

    si = io.StringIO()
    fieldnames = result.get("column_order", list(table[0].keys()))
    writer = csv.DictWriter(si, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(table)

    mem = io.BytesIO(si.getvalue().encode("utf-8-sig"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="pareto_solutions.csv",
    )


# ---- 予測シミュレータ ---------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """単一の変数値セットに対して予測する。"""
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "無効なJSONです"}), 400

    values = body.get("values", {})

    # config はリクエストから直接取得（フロントエンドが保持）
    # フォールバックとしてジョブストアからも取得
    config = body.get("config")
    if config is None:
        job_id = body.get("job_id")
        job = _get_job_snapshot(job_id) if job_id else None
        if job is not None:
            config = job.get("config")
    if config is None:
        return jsonify({"error": "設定が見つかりません。ページを再読み込みして最適化を再実行してください。"}), 400

    try:
        result = predict_single(config, values)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ---- モデルアップロード（カスタムモード用） ----------------------------------

@app.route("/api/upload-model", methods=["POST"])
def api_upload_model():
    """joblib モデルファイルをアップロードする。"""
    if "file" not in request.files:
        return jsonify({"error": "ファイルが選択されていません"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "ファイル名が空です"}), 400

    # 拡張子を検証
    if not f.filename.endswith(".joblib"):
        return jsonify({"error": ".joblib ファイルのみアップロード可能です"}), 400

    safe_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
    save_path = UPLOAD_DIR / safe_name
    f.save(str(save_path))

    try:
        import joblib

        model = joblib.load(str(save_path))
        features = []
        if hasattr(model, "params") and hasattr(model.params, "index"):
            features = model.params.index.tolist()

        # ファイル名から目的変数名（特性名）を自動抽出
        target_name = _extract_target_name(f.filename)

        # 特徴量名から設計変数名を自動抽出
        variable_names = _extract_variable_names(features)

        return jsonify({
            "path": str(save_path),
            "filename": f.filename,
            "features": features,
            "target_name": target_name,
            "variable_names": variable_names,
        })
    except Exception:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "モデルの読み込みに失敗しました。有効な joblib ファイルか確認してください。"}), 400


def _extract_target_name(filename: str) -> str:
    """ファイル名から目的変数名を抽出する。例: ols_特性A.joblib → 特性A"""
    name = filename
    if name.startswith("ols_"):
        name = name[4:]
    if name.endswith(".joblib"):
        name = name[:-7]
    return name


def _extract_variable_names(features: list) -> list:
    """OLS モデルの特徴量名から設計因子（一次項のみ）を抽出する。

    設計因子 = 全回帰モデルの一次項に登場するユニークな変数名。
    二乗項 (VAR^2 (centered)) や交互作用項 (VAR1:VAR2 (centered)) は
    既存の一次項変数から特徴量を構築するため、ここでは抽出しない。
    """
    var_names = set()
    for feat in features:
        if feat == "const":
            continue
        # 二乗項・交互作用項はスキップ（一次項のみ抽出）
        if "^2" in feat and "(centered)" in feat:
            continue
        if ":" in feat and "(centered)" in feat:
            continue
        # 一次項（変数名そのもの）
        var_names.add(feat)
    return sorted(var_names)


def _validate_custom_config(config: dict) -> str | None:
    """カスタムモードの設定を厳密に検証する。不整合があればエラーメッセージを返す。"""
    variables = config.get("variables")
    model_paths = config.get("model_paths")
    targets = config.get("targets")

    if not variables or not isinstance(variables, dict):
        return "設計変数が定義されていません。"
    if not model_paths or not isinstance(model_paths, dict):
        return "回帰モデルがアップロードされていません。"
    if not targets or not isinstance(targets, dict):
        return "目標値が定義されていません。"

    model_keys = set(model_paths.keys())
    target_keys = set(targets.keys())

    # モデルの目的変数と目標値の整合性チェック
    models_without_targets = model_keys - target_keys
    if models_without_targets:
        return (
            f"以下のモデルに対応する目標値が未定義です: "
            f"{', '.join(sorted(models_without_targets))}"
        )

    targets_without_models = target_keys - model_keys
    if targets_without_models:
        return (
            f"以下の目標値に対応するモデルがありません: "
            f"{', '.join(sorted(targets_without_models))}"
        )

    # 各モデルが参照する変数が設計変数に含まれているかチェック
    import joblib

    defined_vars = set(variables.keys())
    for target_name, path in model_paths.items():
        try:
            model = joblib.load(path)
            if hasattr(model, "params") and hasattr(model.params, "index"):
                features = model.params.index.tolist()
                required_vars = set(_extract_variable_names(features))
                missing_vars = required_vars - defined_vars
                if missing_vars:
                    return (
                        f"モデル「{target_name}」が変数 "
                        f"{', '.join(sorted(missing_vars))} を参照していますが、"
                        f"設計変数に定義されていません。"
                    )
        except Exception:
            return f"モデル「{target_name}」の読み込みに失敗しました: {path}"

    # 設計変数の値の検証
    for var_name, bounds in variables.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 3:
            return f"設計変数「{var_name}」のフォーマットが不正です（[下限, 上限, 中心値] が必要）。"
        lo, hi, center = bounds
        if lo >= hi:
            return f"設計変数「{var_name}」の下限 ({lo}) が上限 ({hi}) 以上です。"

    # directions の検証（オプション）
    directions = config.get("directions", {})
    valid_directions = {"target", "maximize", "minimize"}
    for name, direction in directions.items():
        if direction not in valid_directions:
            return (
                f"目的変数「{name}」の方向「{direction}」が不正です。"
                f"有効な値: {', '.join(sorted(valid_directions))}"
            )

    return None


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
