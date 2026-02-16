#!/usr/bin/env python3
"""
アップロード → 自動抽出 → 予測 → 最適化の全パイプライン検証スクリプト。

テスト用の OLS 回帰モデルを3つ作成し、Flask API を通して
一連のフローが正しく動作するか検証する。
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

# =======================================================================
# 1. テスト用 OLS モデルの作成
# =======================================================================

def create_test_models():
    """
    設計因子: A, B, C （一次項）
    目的変数: 硬度, 強度, 密度 の3つの回帰モデルを作成

    各モデルの特徴量:
      硬度: const, A, B, C, A^2 (centered), B^2 (centered), A:B (centered)
      強度: const, A, B, A^2 (centered), A:B (centered)
      密度: const, A, B, C, B:C (centered)
    """
    import statsmodels.api as sm

    np.random.seed(42)
    n = 50

    # 設計因子の値を生成（範囲: A=[10,50], B=[0.5,5.0], C=[100,300]）
    A = np.random.uniform(10, 50, n)
    B = np.random.uniform(0.5, 5.0, n)
    C = np.random.uniform(100, 300, n)

    # 中心値
    A_c, B_c, C_c = 30.0, 2.75, 200.0

    os.makedirs("test_models", exist_ok=True)
    models_info = {}

    # --- 硬度モデル ---
    # 特徴量: const, A, B, C, A^2(centered), B^2(centered), A:B(centered)
    df_h = pd.DataFrame({
        "const": np.ones(n),
        "A": A,
        "B": B,
        "C": C,
        "A^2 (centered)": (A - A_c) ** 2,
        "B^2 (centered)": (B - B_c) ** 2,
        "A:B (centered)": (A - A_c) * (B - B_c),
    })
    # 真の係数
    true_params_h = np.array([50.0, 1.2, -3.5, 0.05, -0.01, 0.8, 0.15])
    y_h = df_h.values @ true_params_h + np.random.normal(0, 0.5, n)
    model_h = sm.OLS(y_h, df_h).fit()
    path_h = "test_models/ols_硬度.joblib"
    joblib.dump(model_h, path_h)
    models_info["硬度"] = {
        "path": os.path.abspath(path_h),
        "features": df_h.columns.tolist(),
        "true_params": true_params_h.tolist(),
    }
    print(f"  硬度: features={df_h.columns.tolist()}")
    print(f"         true_params={true_params_h.tolist()}")
    print(f"         fitted_params={model_h.params.values.tolist()}")

    # --- 強度モデル ---
    # 特徴量: const, A, B, A^2(centered), A:B(centered)
    df_s = pd.DataFrame({
        "const": np.ones(n),
        "A": A,
        "B": B,
        "A^2 (centered)": (A - A_c) ** 2,
        "A:B (centered)": (A - A_c) * (B - B_c),
    })
    true_params_s = np.array([100.0, 2.0, 5.0, -0.02, 0.3])
    y_s = df_s.values @ true_params_s + np.random.normal(0, 1.0, n)
    model_s = sm.OLS(y_s, df_s).fit()
    path_s = "test_models/ols_強度.joblib"
    joblib.dump(model_s, path_s)
    models_info["強度"] = {
        "path": os.path.abspath(path_s),
        "features": df_s.columns.tolist(),
        "true_params": true_params_s.tolist(),
    }
    print(f"  強度: features={df_s.columns.tolist()}")
    print(f"         true_params={true_params_s.tolist()}")
    print(f"         fitted_params={model_s.params.values.tolist()}")

    # --- 密度モデル ---
    # 特徴量: const, A, B, C, B:C(centered)
    df_d = pd.DataFrame({
        "const": np.ones(n),
        "A": A,
        "B": B,
        "C": C,
        "B:C (centered)": (B - B_c) * (C - C_c),
    })
    true_params_d = np.array([7.0, 0.01, -0.2, 0.005, 0.001])
    y_d = df_d.values @ true_params_d + np.random.normal(0, 0.1, n)
    model_d = sm.OLS(y_d, df_d).fit()
    path_d = "test_models/ols_密度.joblib"
    joblib.dump(model_d, path_d)
    models_info["密度"] = {
        "path": os.path.abspath(path_d),
        "features": df_d.columns.tolist(),
        "true_params": true_params_d.tolist(),
    }
    print(f"  密度: features={df_d.columns.tolist()}")
    print(f"         true_params={true_params_d.tolist()}")
    print(f"         fitted_params={model_d.params.values.tolist()}")

    return models_info


# =======================================================================
# 2. Flask API テスト
# =======================================================================

def test_upload_model(client, filepath, filename):
    """モデルアップロードをテストする。"""
    with open(filepath, "rb") as f:
        data = {"file": (f, filename)}
        resp = client.post(
            "/api/upload-model",
            data=data,
            content_type="multipart/form-data",
        )
    return resp.get_json(), resp.status_code


def test_predict(client, config, values):
    """予測APIをテストする。"""
    resp = client.post(
        "/api/predict",
        data=json.dumps({"config": config, "values": values}),
        content_type="application/json",
    )
    return resp.get_json(), resp.status_code


def test_optimize(client, config):
    """最適化APIをテストする。"""
    resp = client.post(
        "/api/optimize",
        data=json.dumps(config),
        content_type="application/json",
    )
    return resp.get_json(), resp.status_code


def test_progress(client, job_id):
    """進捗確認APIをテストする。"""
    resp = client.get(f"/api/progress/{job_id}")
    return resp.get_json(), resp.status_code


def test_results(client, job_id):
    """結果取得APIをテストする。"""
    resp = client.get(f"/api/results/{job_id}")
    return resp.get_json(), resp.status_code


# =======================================================================
# メイン
# =======================================================================

def main():
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} -- {detail}")
            failed += 1

    # --- モデル作成 ---
    print("=" * 60)
    print("1. テスト用 OLS モデルの作成")
    print("=" * 60)
    models_info = create_test_models()
    print()

    # --- Flask テストクライアント ---
    from app import app
    app.config["TESTING"] = True
    client = app.test_client()

    # === TEST: モデルアップロード ===
    print("=" * 60)
    print("2. モデルアップロード & 自動抽出テスト")
    print("=" * 60)

    uploaded = []
    for target_name, info in models_info.items():
        filename = f"ols_{target_name}.joblib"
        data, status = test_upload_model(client, info["path"], filename)
        check(
            f"アップロード成功: {filename}",
            status == 200 and "error" not in data,
            f"status={status}, data={data}",
        )
        if status == 200:
            # target_name が正しく抽出されたか
            check(
                f"target_name 抽出: {filename} → {data.get('target_name')}",
                data.get("target_name") == target_name,
                f"expected={target_name}, got={data.get('target_name')}",
            )
            # variable_names が一次項のみか
            expected_vars = sorted(
                [f for f in info["features"] if f != "const"
                 and "^2" not in f and ":" not in f]
            )
            check(
                f"variable_names 抽出 (一次項のみ): {data.get('variable_names')}",
                data.get("variable_names") == expected_vars,
                f"expected={expected_vars}, got={data.get('variable_names')}",
            )
            # features が全特徴量を含むか
            check(
                f"features 全量返却: {len(data.get('features', []))} 特徴量",
                set(data.get("features", [])) == set(info["features"]),
                f"expected={info['features']}, got={data.get('features')}",
            )
            uploaded.append(data)

    print()

    # === TEST: 予測 ===
    print("=" * 60)
    print("3. 予測 API テスト")
    print("=" * 60)

    # テスト用 config を構築
    config = {
        "mode": "custom",
        "variables": {
            "A": [10, 50, 30.0],
            "B": [0.5, 5.0, 2.75],
            "C": [100, 300, 200.0],
        },
        "model_paths": {u["target_name"]: u["path"] for u in uploaded},
        "targets": {
            "硬度": 70.0,
            "強度": 150.0,
            "密度": 8.0,
        },
        "deviation_mode": "absolute",
        "aggregation_mode": "full",
        "pop_size": 50,
        "n_gen": 30,
        "seed": 42,
    }

    # 中心点での予測（A=30, B=2.75, C=200）
    center_values = {"A": 30.0, "B": 2.75, "C": 200.0}
    pred_result, status = test_predict(client, config, center_values)
    check(
        "予測 API 成功 (中心点)",
        status == 200 and "error" not in pred_result,
        f"status={status}, data={pred_result}",
    )
    if status == 200 and "predictions" in pred_result:
        for name, info in pred_result["predictions"].items():
            print(f"    {name}: predicted={info['predicted']}, target={info['target']}, dev={info['abs_deviation']}")

    # 手動で計算して予測値を検証
    print("\n  --- 手動計算との比較 ---")
    for target_name, minfo in models_info.items():
        model = joblib.load(minfo["path"])
        feats = model.params.index.tolist()
        params = model.params.values

        # 手動で特徴量を構築
        A_val, B_val, C_val = 30.0, 2.75, 200.0
        A_c, B_c, C_c = 30.0, 2.75, 200.0
        feat_map = {
            "const": 1.0,
            "A": A_val,
            "B": B_val,
            "C": C_val,
            "A^2 (centered)": (A_val - A_c) ** 2,
            "B^2 (centered)": (B_val - B_c) ** 2,
            "A:B (centered)": (A_val - A_c) * (B_val - B_c),
            "B:C (centered)": (B_val - B_c) * (C_val - C_c),
        }
        feat_vals = np.array([feat_map.get(f, 0.0) for f in feats])
        manual_pred = float(feat_vals @ params)

        api_pred = pred_result["predictions"][target_name]["predicted"] if target_name in pred_result.get("predictions", {}) else None
        match = api_pred is not None and abs(api_pred - manual_pred) < 1e-4
        check(
            f"予測一致 {target_name}: API={api_pred}, 手動={round(manual_pred, 6)}",
            match,
            f"diff={abs(api_pred - manual_pred) if api_pred else 'N/A'}",
        )

    # 非中心点での予測テスト（A=40, B=3.5, C=250）
    off_center = {"A": 40.0, "B": 3.5, "C": 250.0}
    pred2, status2 = test_predict(client, config, off_center)
    check(
        "予測 API 成功 (非中心点)",
        status2 == 200 and "error" not in pred2,
        f"status={status2}",
    )
    if status2 == 200 and "predictions" in pred2:
        for target_name, minfo in models_info.items():
            model = joblib.load(minfo["path"])
            feats = model.params.index.tolist()
            params = model.params.values

            A_val, B_val, C_val = 40.0, 3.5, 250.0
            feat_map = {
                "const": 1.0,
                "A": A_val,
                "B": B_val,
                "C": C_val,
                "A^2 (centered)": (A_val - 30.0) ** 2,
                "B^2 (centered)": (B_val - 2.75) ** 2,
                "A:B (centered)": (A_val - 30.0) * (B_val - 2.75),
                "B:C (centered)": (B_val - 2.75) * (C_val - 200.0),
            }
            feat_vals = np.array([feat_map.get(f, 0.0) for f in feats])
            manual_pred = float(feat_vals @ params)
            api_pred = pred2["predictions"][target_name]["predicted"]
            match = abs(api_pred - manual_pred) < 1e-4
            check(
                f"予測一致 (非中心) {target_name}: API={api_pred}, 手動={round(manual_pred, 6)}",
                match,
                f"diff={abs(api_pred - manual_pred)}",
            )

    print()

    # === TEST: バリデーション ===
    print("=" * 60)
    print("4. バリデーションテスト")
    print("=" * 60)

    # 4a. モデル数 > 目標値数（目標値が足りない）
    bad_config_1 = {**config, "targets": {"硬度": 70.0, "強度": 150.0}}  # 密度が抜けている
    resp1, status1 = test_optimize(client, bad_config_1)
    check(
        "目標値不足で拒否",
        status1 == 400 and "密度" in resp1.get("error", ""),
        f"status={status1}, error={resp1.get('error', '')}",
    )

    # 4b. 目標値 > モデル数（余分な目標値）
    bad_config_2 = {**config, "targets": {"硬度": 70.0, "強度": 150.0, "密度": 8.0, "粘度": 5.0}}
    resp2, status2 = test_optimize(client, bad_config_2)
    check(
        "余分な目標値で拒否",
        status2 == 400 and "粘度" in resp2.get("error", ""),
        f"status={status2}, error={resp2.get('error', '')}",
    )

    # 4c. 設計変数が不足（Cを削除）
    bad_config_3 = {
        **config,
        "variables": {
            "A": [10, 50, 30.0],
            "B": [0.5, 5.0, 2.75],
            # C が不足
        },
    }
    resp3, status3 = test_optimize(client, bad_config_3)
    check(
        "設計変数不足で拒否 (C が不足)",
        status3 == 400 and "C" in resp3.get("error", ""),
        f"status={status3}, error={resp3.get('error', '')}",
    )

    # 4d. 下限 >= 上限
    bad_config_4 = {
        **config,
        "variables": {
            "A": [50, 10, 30.0],  # 下限 > 上限
            "B": [0.5, 5.0, 2.75],
            "C": [100, 300, 200.0],
        },
    }
    resp4, status4 = test_optimize(client, bad_config_4)
    check(
        "下限>=上限で拒否",
        status4 == 400 and "下限" in resp4.get("error", ""),
        f"status={status4}, error={resp4.get('error', '')}",
    )

    # 4e. 正常な config は通る
    resp_ok, status_ok = test_optimize(client, config)
    check(
        "正常 config は受理",
        status_ok == 200 and "job_id" in resp_ok,
        f"status={status_ok}, data={resp_ok}",
    )

    print()

    # === TEST: 最適化の実行 ===
    print("=" * 60)
    print("5. 最適化実行テスト")
    print("=" * 60)

    if status_ok == 200:
        job_id = resp_ok["job_id"]
        print(f"  job_id = {job_id}")

        # 完了まで待機
        max_wait = 120
        start = time.time()
        final_status = None
        while time.time() - start < max_wait:
            prog, _ = test_progress(client, job_id)
            final_status = prog.get("status")
            if final_status in ("completed", "error"):
                break
            time.sleep(1)

        check(
            "最適化が完了",
            final_status == "completed",
            f"status={final_status}",
        )

        if final_status == "completed":
            result, _ = test_results(client, job_id)

            n_solutions = result.get("n_solutions", 0)
            check(f"パレート解が存在: {n_solutions} 解", n_solutions > 0)

            var_names = result.get("var_names", [])
            check(
                f"設計変数名が正しい: {var_names}",
                set(var_names) == {"A", "B", "C"},
                f"got={var_names}",
            )

            obj_names = result.get("obj_names", [])
            check(
                f"目的関数名が正しい: {obj_names}",
                set(obj_names) == {"dev_硬度", "dev_強度", "dev_密度"},
                f"got={obj_names}",
            )

            # predictions が含まれているか
            preds = result.get("predictions", {})
            check(
                f"予測値が含まれる: {list(preds.keys())}",
                set(preds.keys()) == {"pred_硬度", "pred_強度", "pred_密度"},
                f"got={list(preds.keys())}",
            )

            # deviations が含まれているか
            devs = result.get("deviations", {})
            check(
                f"偏差値が含まれる: {list(devs.keys())}",
                set(devs.keys()) == {"dev_硬度", "dev_強度", "dev_密度"},
                f"got={list(devs.keys())}",
            )

            # テーブルデータの検証
            table = result.get("table", [])
            if table:
                first_row = table[0]
                expected_cols = set(var_names) | {f"pred_{t}" for t in ["硬度", "強度", "密度"]} | set(obj_names)
                actual_cols = set(first_row.keys())
                check(
                    f"テーブル列が正しい ({len(actual_cols)} 列)",
                    expected_cols == actual_cols,
                    f"expected={sorted(expected_cols)}, got={sorted(actual_cols)}",
                )

                # パレート解の各行で予測値を手動検証（最初の3行）
                print("\n  --- パレート解の予測値を手動検証 ---")
                for row_idx in range(min(3, len(table))):
                    row = table[row_idx]
                    A_val = row["A"]
                    B_val = row["B"]
                    C_val = row["C"]

                    for target_name in ["硬度", "強度", "密度"]:
                        model = joblib.load(models_info[target_name]["path"])
                        feats = model.params.index.tolist()
                        params = model.params.values

                        feat_map = {
                            "const": 1.0,
                            "A": A_val,
                            "B": B_val,
                            "C": C_val,
                            "A^2 (centered)": (A_val - 30.0) ** 2,
                            "B^2 (centered)": (B_val - 2.75) ** 2,
                            "A:B (centered)": (A_val - 30.0) * (B_val - 2.75),
                            "B:C (centered)": (B_val - 2.75) * (C_val - 200.0),
                        }
                        feat_vals = np.array([feat_map.get(f, 0.0) for f in feats])
                        manual_pred = float(feat_vals @ params)
                        api_pred = row[f"pred_{target_name}"]
                        diff = abs(api_pred - manual_pred)
                        check(
                            f"  解#{row_idx} {target_name}: API={api_pred}, 手動={round(manual_pred, 6)}, diff={diff:.2e}",
                            diff < 1e-3,
                            f"diff too large: {diff}",
                        )

    print()

    # === 結果サマリー ===
    print("=" * 60)
    total = passed + failed
    print(f"結果: {passed}/{total} テスト PASS, {failed} FAIL")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
