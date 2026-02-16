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

    job_id = body.get("job_id")
    values = body.get("values", {})

    job = _get_job_snapshot(job_id)
    if job is None:
        return jsonify({"error": "ジョブが見つかりません"}), 404
    config = job.get("config")
    if config is None:
        return jsonify({"error": "ジョブ設定が見つかりません"}), 400

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
        return jsonify({
            "path": str(save_path),
            "filename": f.filename,
            "features": features,
        })
    except Exception:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "モデルの読み込みに失敗しました。有効な joblib ファイルか確認してください。"}), 400


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
