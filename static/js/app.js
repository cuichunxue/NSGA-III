/* ===================================================================
   NSGA-III 最適化ツール – フロントエンド
   =================================================================== */

(function () {
    "use strict";

    // ----- state -------------------------------------------------------
    const S = {
        mode: "demo",           // "demo" | "custom"
        selectedProblem: null,  // demo問題キー
        demoProblems: {},
        uploadedModels: [],     // [{path, filename, features, target_name, variable_names}]
        jobId: null,
        pollTimer: null,
        resultData: null,
        sortCol: null,
        sortAsc: true,
        predictRefRow: null,  // 予測比較用のパレート解参照値
    };

    // ----- DOM refs ----------------------------------------------------
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // ----- 初期化 -------------------------------------------------------
    document.addEventListener("DOMContentLoaded", init);

    function init() {
        loadDemoProblems();
        bindNavigation();
        bindModeToggle();
        bindCustomForm();
        bindPresets();
        bindTabs();
        bindRunControls();
        bindResultControls();
    }

    // ----- デモ問題ロード ------------------------------------------------
    async function loadDemoProblems() {
        try {
            const res = await fetch("/api/demo-problems");
            S.demoProblems = await res.json();
            renderDemoCards();
        } catch (e) {
            console.error("デモ問題の取得に失敗:", e);
        }
    }

    function renderDemoCards() {
        const grid = $("#demo-cards");
        grid.innerHTML = "";
        for (const [key, info] of Object.entries(S.demoProblems)) {
            const card = document.createElement("div");
            card.className = "demo-card";
            card.dataset.key = key;
            card.innerHTML = `
                <h3>${escapeHtml(info.name)}</h3>
                <p>${escapeHtml(info.description)}</p>
                <span class="badge">${Number(info.n_obj)}目的 / ${Number(info.n_var)}変数</span>
            `;
            card.addEventListener("click", () => selectDemoCard(key));
            grid.appendChild(card);
        }
    }

    function selectDemoCard(key) {
        S.selectedProblem = key;
        $$(".demo-card").forEach((c) => c.classList.toggle("selected", c.dataset.key === key));
        $("#btn-to-step2").disabled = false;
    }

    // ----- モード切替 ----------------------------------------------------
    function bindModeToggle() {
        $$(".btn-mode").forEach((btn) => {
            btn.addEventListener("click", () => {
                S.mode = btn.dataset.mode;
                $$(".btn-mode").forEach((b) => b.classList.toggle("active", b === btn));
                toggleMode();
            });
        });
    }

    function toggleMode() {
        const isDemo = S.mode === "demo";
        $("#demo-section").classList.toggle("hidden", !isDemo);
        $("#custom-section").classList.toggle("hidden", isDemo);
        // カスタムモードでは常に次へ有効
        if (!isDemo) {
            $("#btn-to-step2").disabled = false;
        } else {
            $("#btn-to-step2").disabled = !S.selectedProblem;
        }
    }

    // ----- カスタムフォーム -----------------------------------------------
    function bindCustomForm() {
        // 変数行の追加
        $("#btn-add-var").addEventListener("click", () => {
            const idx = $$("#var-table-body tr").length + 1;
            const row = document.createElement("tr");
            row.innerHTML = `
                <td><input type="text" class="var-name" value="x${idx}" placeholder="変数名"></td>
                <td><input type="number" class="var-lo" value="0" step="any"></td>
                <td><input type="number" class="var-hi" value="1" step="any"></td>
                <td><input type="number" class="var-center" value="0.5" step="any"></td>
                <td><button class="btn-icon btn-remove-var" title="削除">&times;</button></td>
            `;
            $("#var-table-body").appendChild(row);
            bindRemoveButtons();
        });

        // 目標行の追加
        $("#btn-add-target").addEventListener("click", () => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td><input type="text" class="tgt-name" placeholder="特性名"></td>
                <td>
                    <select class="tgt-direction">
                        <option value="target">目標値に近づける</option>
                        <option value="maximize">最大化</option>
                        <option value="minimize">最小化</option>
                    </select>
                </td>
                <td><input type="number" class="tgt-val" value="0" step="any"></td>
                <td><button class="btn-icon btn-remove-tgt" title="削除">&times;</button></td>
            `;
            $("#target-table-body").appendChild(row);
            bindRemoveButtons();
            bindDirectionToggles();
        });

        // モデルアップロード
        $("#model-file-input").addEventListener("change", handleModelUpload);

        bindRemoveButtons();
        bindDirectionToggles();
    }

    function bindDirectionToggles() {
        $$("#target-table-body .tgt-direction").forEach((sel) => {
            sel.onchange = () => {
                const valInput = sel.closest("tr").querySelector(".tgt-val");
                const isTarget = sel.value === "target";
                valInput.disabled = !isTarget;
                if (!isTarget) valInput.value = "";
            };
        });
    }

    function bindRemoveButtons() {
        $$(".btn-remove-var").forEach((btn) => {
            btn.onclick = () => {
                if ($$("#var-table-body tr").length > 1) btn.closest("tr").remove();
            };
        });
        $$(".btn-remove-tgt").forEach((btn) => {
            btn.onclick = () => {
                if ($$("#target-table-body tr").length > 1) btn.closest("tr").remove();
            };
        });
    }

    async function handleModelUpload(e) {
        const files = e.target.files;
        for (const file of files) {
            const fd = new FormData();
            fd.append("file", file);
            try {
                const res = await fetch("/api/upload-model", { method: "POST", body: fd });
                const data = await res.json();
                if (data.error) {
                    alert("アップロードエラー: " + data.error);
                    continue;
                }
                S.uploadedModels.push(data);
            } catch (err) {
                alert("アップロードに失敗しました: " + err.message);
            }
        }
        e.target.value = "";
        renderUploadedModels();
        autoPopulateFromModels();
        clearCustomErrors();
    }

    function escapeHtml(str) {
        const d = document.createElement("div");
        d.textContent = str;
        return d.innerHTML;
    }

    function renderUploadedModels() {
        const el = $("#uploaded-models");
        el.innerHTML = S.uploadedModels
            .map(
                (m, i) => `
            <div class="uploaded-item">
                <span>${escapeHtml(m.filename)}
                    <small class="hint">[目的変数: ${escapeHtml(m.target_name || "不明")}] (${m.features.length} 特徴量, ${(m.variable_names || []).length} 設計変数)</small>
                </span>
                <button class="btn-icon" data-idx="${i}" title="削除">&times;</button>
            </div>
        `
            )
            .join("");
        el.querySelectorAll(".btn-icon").forEach((btn) => {
            btn.addEventListener("click", () => {
                S.uploadedModels.splice(parseInt(btn.dataset.idx), 1);
                renderUploadedModels();
                autoPopulateFromModels();
                clearCustomErrors();
            });
        });
    }

    /**
     * アップロード済みモデルから設計変数テーブルと目標値テーブルを自動補完する。
     * 既存の手動入力を尊重しつつ、不足分を追加する。
     */
    function autoPopulateFromModels() {
        if (S.uploadedModels.length === 0) return;

        // 全モデルから設計変数名を収集
        const allVarNames = new Set();
        S.uploadedModels.forEach((m) => {
            (m.variable_names || []).forEach((v) => allVarNames.add(v));
        });

        // 全モデルから目的変数名を収集
        const allTargetNames = new Set();
        S.uploadedModels.forEach((m) => {
            if (m.target_name) allTargetNames.add(m.target_name);
        });

        // --- 設計変数テーブルの自動補完 ---
        const existingVarNames = new Set();
        $$("#var-table-body tr").forEach((row) => {
            const name = row.querySelector(".var-name").value.trim();
            if (name) existingVarNames.add(name);
        });

        // 既存行が空のデフォルト値のみなら全クリアして再構築
        const existingRows = $$("#var-table-body tr");
        const onlyDefaults = existingRows.length === 1
            && existingRows[0].querySelector(".var-name").value.trim() === "x1"
            && !allVarNames.has("x1");
        if (onlyDefaults) {
            $("#var-table-body").innerHTML = "";
            existingVarNames.clear();
        }

        // 不足している変数を追加
        allVarNames.forEach((varName) => {
            if (!existingVarNames.has(varName)) {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td><input type="text" class="var-name" value="${escapeHtml(varName)}" placeholder="変数名"></td>
                    <td><input type="number" class="var-lo" value="0" step="any"></td>
                    <td><input type="number" class="var-hi" value="1" step="any"></td>
                    <td><input type="number" class="var-center" value="0.5" step="any"></td>
                    <td><button class="btn-icon btn-remove-var" title="削除">&times;</button></td>
                `;
                $("#var-table-body").appendChild(row);
            }
        });

        // --- 目標値テーブルの自動補完 ---
        const existingTargetNames = new Set();
        $$("#target-table-body tr").forEach((row) => {
            const name = row.querySelector(".tgt-name").value.trim();
            if (name) existingTargetNames.add(name);
        });

        // 既存行が空のデフォルトのみなら全クリアして再構築
        const existingTargetRows = $$("#target-table-body tr");
        const onlyDefaultTargets = existingTargetRows.length === 1
            && existingTargetRows[0].querySelector(".tgt-name").value.trim() === "";
        if (onlyDefaultTargets) {
            $("#target-table-body").innerHTML = "";
            existingTargetNames.clear();
        }

        // 不足している目標値を追加
        allTargetNames.forEach((tgtName) => {
            if (!existingTargetNames.has(tgtName)) {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td><input type="text" class="tgt-name" value="${escapeHtml(tgtName)}" placeholder="特性名"></td>
                    <td>
                        <select class="tgt-direction">
                            <option value="target">目標値に近づける</option>
                            <option value="maximize">最大化</option>
                            <option value="minimize">最小化</option>
                        </select>
                    </td>
                    <td><input type="number" class="tgt-val" value="0" step="any"></td>
                    <td><button class="btn-icon btn-remove-tgt" title="削除">&times;</button></td>
                `;
                $("#target-table-body").appendChild(row);
            }
        });

        bindRemoveButtons();
        bindDirectionToggles();
    }

    /**
     * カスタムモードのバリデーション。
     * エラーがあればメッセージ配列を返し、問題なければ空配列を返す。
     */
    function validateCustomConfig() {
        const errors = [];

        // モデルが1つもアップロードされていない
        if (S.uploadedModels.length === 0) {
            errors.push("回帰モデルが1つもアップロードされていません。.joblib ファイルをアップロードしてください。");
            return errors;
        }

        // モデルから目的変数名を収集
        const modelTargetNames = new Set();
        S.uploadedModels.forEach((m) => {
            if (m.target_name) modelTargetNames.add(m.target_name);
        });

        // モデルから設計変数名を収集
        const modelVarNames = new Set();
        S.uploadedModels.forEach((m) => {
            (m.variable_names || []).forEach((v) => modelVarNames.add(v));
        });

        // ユーザーが定義した設計変数を収集
        const definedVars = {};
        $$("#var-table-body tr").forEach((row) => {
            const name = row.querySelector(".var-name").value.trim();
            const lo = parseFloat(row.querySelector(".var-lo").value);
            const hi = parseFloat(row.querySelector(".var-hi").value);
            if (name) {
                definedVars[name] = { lo, hi };
            }
        });

        // ユーザーが定義した目標値を収集
        const definedTargets = new Set();
        const definedDirections = {};
        $$("#target-table-body tr").forEach((row) => {
            const name = row.querySelector(".tgt-name").value.trim();
            const dir = row.querySelector(".tgt-direction").value;
            const val = row.querySelector(".tgt-val").value;
            if (name) {
                definedTargets.add(name);
                definedDirections[name] = { direction: dir, value: val };
            }
        });

        // チェック1: モデルの目的変数に対応する目標値が全て定義されているか
        modelTargetNames.forEach((tgt) => {
            if (!definedTargets.has(tgt)) {
                errors.push(`モデル「${tgt}」に対応する目標値が未定義です。目標値テーブルに「${tgt}」を追加してください。`);
            }
        });

        // チェック2: 定義された目標値に対応するモデルが全てあるか
        definedTargets.forEach((tgt) => {
            if (!modelTargetNames.has(tgt)) {
                errors.push(`目標値「${tgt}」に対応するモデルがアップロードされていません。`);
            }
        });

        // チェック3: モデルが参照する設計変数が全て定義されているか
        modelVarNames.forEach((varName) => {
            if (!(varName in definedVars)) {
                errors.push(`モデルが変数「${varName}」を参照していますが、設計変数テーブルに定義されていません。`);
            }
        });

        // チェック4: 設計変数の下限 < 上限
        for (const [name, bounds] of Object.entries(definedVars)) {
            if (isNaN(bounds.lo) || isNaN(bounds.hi)) {
                errors.push(`設計変数「${name}」の上下限が数値ではありません。`);
            } else if (bounds.lo >= bounds.hi) {
                errors.push(`設計変数「${name}」の下限 (${bounds.lo}) が上限 (${bounds.hi}) 以上です。`);
            }
        }

        // チェック5: 設計変数が0個
        if (Object.keys(definedVars).length === 0) {
            errors.push("設計変数が1つも定義されていません。");
        }

        // チェック6: 目標値が0個
        if (definedTargets.size === 0) {
            errors.push("目標値が1つも定義されていません。");
        }

        // チェック7: 「目標値に近づける」方向で目標値が空
        for (const [name, info] of Object.entries(definedDirections)) {
            if (info.direction === "target" && (info.value === "" || isNaN(parseFloat(info.value)))) {
                errors.push(`「${name}」の方向が「目標値に近づける」ですが、目標値が未入力です。`);
            }
        }

        return errors;
    }

    function clearCustomErrors() {
        const el = $("#custom-validation-errors");
        if (el) {
            el.classList.add("hidden");
            el.innerHTML = "";
        }
    }

    function showCustomErrors(errors) {
        const el = $("#custom-validation-errors");
        if (!el) return;
        if (errors.length === 0) {
            el.classList.add("hidden");
            el.innerHTML = "";
            return;
        }
        el.classList.remove("hidden");
        el.innerHTML = `
            <div class="error-box">
                <strong>設定エラー（${errors.length}件）</strong>
                <ul>${errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>
            </div>
        `;
    }

    // ----- プリセット ------------------------------------------------
    function bindPresets() {
        $$(".btn-preset").forEach((btn) => {
            btn.addEventListener("click", () => {
                $("#pop-size").value = btn.dataset.pop;
                $("#n-gen").value = btn.dataset.gen;
                $$(".btn-preset").forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
            });
        });
    }

    // ----- タブ切替 ------------------------------------------------
    function bindTabs() {
        $$(".tab").forEach((tab) => {
            tab.addEventListener("click", () => {
                $$(".tab").forEach((t) => t.classList.remove("active"));
                $$(".tab-content").forEach((c) => c.classList.remove("active"));
                tab.classList.add("active");
                $(`#tab-${tab.dataset.tab}`).classList.add("active");
            });
        });
    }

    // ----- ステップナビゲーション ----------------------------------------
    function bindNavigation() {
        $("#btn-to-step2").addEventListener("click", () => {
            // カスタムモードの場合はバリデーション実行
            if (S.mode === "custom") {
                const errors = validateCustomConfig();
                if (errors.length > 0) {
                    showCustomErrors(errors);
                    return;
                }
                clearCustomErrors();
            }
            goToStep(2);
        });
        $("#btn-to-step3").addEventListener("click", () => goToStep(3));
        $("#btn-back-step1").addEventListener("click", () => goToStep(1));
        $("#btn-back-step2").addEventListener("click", () => goToStep(2));
        $("#btn-new").addEventListener("click", () => {
            S.jobId = null;
            S.resultData = null;
            goToStep(1);
        });
    }

    function goToStep(n) {
        // パネル
        for (let i = 1; i <= 4; i++) {
            const panel = $(`#step${i}-panel`);
            if (panel) panel.classList.toggle("hidden", i !== n);
        }
        // ステッパー
        $$(".step").forEach((s) => {
            const stepN = parseInt(s.dataset.step);
            s.classList.toggle("active", stepN === n);
            s.classList.toggle("done", stepN < n);
        });
        // Step 3 表示時にサマリー構築
        if (n === 3) buildRunSummary();
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    // ----- 実行サマリー構築 -----------------------------------------
    function buildRunSummary() {
        const pop = $("#pop-size").value;
        const gen = $("#n-gen").value;
        const seed = $("#seed").value;

        let problemDesc = "";
        if (S.mode === "demo") {
            const info = S.demoProblems[S.selectedProblem];
            problemDesc = `<strong>問題:</strong> ${info.name} (${info.n_obj}目的, ${info.n_var}変数)`;
        } else {
            const nVar = $$("#var-table-body tr").length;
            const nModel = S.uploadedModels.length;
            problemDesc = `<strong>問題:</strong> カスタム OLS (変数 ${nVar}個, モデル ${nModel}個)`;
        }

        $("#run-summary").innerHTML = `
            ${problemDesc}<br>
            <strong>母集団:</strong> ${pop} &nbsp; <strong>世代:</strong> ${gen} &nbsp; <strong>シード:</strong> ${seed}
        `;

        // 進捗・エラー表示リセット
        $("#progress-area").classList.add("hidden");
        $("#error-area").classList.add("hidden");
        $("#btn-start").disabled = false;
        $("#btn-start").textContent = "最適化を開始";
    }

    // ----- 実行制御 --------------------------------------------------
    function bindRunControls() {
        $("#btn-start").addEventListener("click", startOptimization);
    }

    function bindResultControls() {
        $("#btn-download").addEventListener("click", downloadCSV);
        $("#table-search").addEventListener("input", filterTable);
    }

    async function startOptimization() {
        const btn = $("#btn-start");
        btn.disabled = true;
        btn.textContent = "送信中...";

        const config = buildConfig();

        try {
            const res = await fetch("/api/optimize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(config),
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            S.jobId = data.job_id;
            S.config = config;
            btn.textContent = "実行中...";
            $("#progress-area").classList.remove("hidden");
            $("#progress-gen-total").textContent = config.n_gen;
            startPolling();
        } catch (err) {
            btn.disabled = false;
            btn.textContent = "最適化を開始";
            showError(err.message);
        }
    }

    function buildConfig() {
        const pop = parseInt($("#pop-size").value);
        const gen = parseInt($("#n-gen").value);
        const seed = parseInt($("#seed").value);

        if (S.mode === "demo") {
            return {
                mode: "demo",
                problem: S.selectedProblem,
                pop_size: pop,
                n_gen: gen,
                seed: seed,
            };
        }

        // カスタムモード
        const variables = {};
        $$("#var-table-body tr").forEach((row) => {
            const name = row.querySelector(".var-name").value.trim();
            const lo = parseFloat(row.querySelector(".var-lo").value);
            const hi = parseFloat(row.querySelector(".var-hi").value);
            const center = parseFloat(row.querySelector(".var-center").value);
            if (name) variables[name] = [lo, hi, center];
        });

        const targets = {};
        const directions = {};
        $$("#target-table-body tr").forEach((row) => {
            const name = row.querySelector(".tgt-name").value.trim();
            const val = parseFloat(row.querySelector(".tgt-val").value);
            const dir = row.querySelector(".tgt-direction").value;
            if (name) {
                targets[name] = dir === "target" ? val : 0;
                directions[name] = dir;
            }
        });

        const modelPaths = {};
        S.uploadedModels.forEach((m) => {
            // サーバーが解析した target_name を使用（整合性保証）
            const key = m.target_name || m.filename.replace(/^ols_/, "").replace(/\.joblib$/, "");
            modelPaths[key] = m.path;
        });

        const aggMode = document.querySelector('input[name="agg-mode"]:checked').value;
        const devMode = document.querySelector('input[name="dev-mode"]:checked').value;

        return {
            mode: "custom",
            variables,
            model_paths: modelPaths,
            targets,
            directions,
            aggregation_mode: aggMode,
            deviation_mode: devMode,
            pop_size: pop,
            n_gen: gen,
            seed: seed,
        };
    }

    // ----- ポーリング --------------------------------------------------
    function startPolling() {
        if (S.pollTimer) clearInterval(S.pollTimer);
        S.pollTimer = setInterval(pollProgress, 800);
    }

    async function pollProgress() {
        if (!S.jobId) return;
        try {
            const res = await fetch(`/api/progress/${S.jobId}`);
            const data = await res.json();

            $("#progress-bar").style.width = data.progress + "%";
            $("#progress-text").textContent = `${data.progress}%`;
            $("#progress-gen").textContent = data.current_gen;
            $("#progress-gen-total").textContent = data.total_gen;
            if (data.elapsed != null) {
                $("#progress-elapsed").textContent = formatElapsed(data.elapsed);
            }

            if (data.status === "completed") {
                clearInterval(S.pollTimer);
                S.pollTimer = null;
                await loadResults();
            } else if (data.status === "error") {
                clearInterval(S.pollTimer);
                S.pollTimer = null;
                showError(data.error || "不明なエラー");
                $("#btn-start").disabled = false;
                $("#btn-start").textContent = "再実行";
            }
        } catch (e) {
            // ネットワークエラー時はリトライを続ける
        }
    }

    function formatElapsed(sec) {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return m > 0 ? `${m}分${s}秒` : `${s}秒`;
    }

    // ----- 結果ロード ---------------------------------------------------
    async function loadResults() {
        try {
            const res = await fetch(`/api/results/${S.jobId}`);
            S.resultData = await res.json();
            if (S.resultData.error) throw new Error(S.resultData.error);
            goToStep(4);
            renderResults();
        } catch (e) {
            showError("結果の読み込みに失敗しました: " + e.message);
        }
    }

    // ----- 結果レンダリング ----------------------------------------------
    function renderResults() {
        const d = S.resultData;
        try { renderSummary(d); } catch (e) { console.error("renderSummary:", e); }
        try { renderParetoChart(d); } catch (e) { console.error("renderParetoChart:", e); }
        try { renderParallelChart(d); } catch (e) { console.error("renderParallelChart:", e); }
        try { renderTable(d); } catch (e) { console.error("renderTable:", e); }
        try { renderPredictInputs(d); } catch (e) { console.error("renderPredictInputs:", e); }
    }

    function renderSummary(d) {
        const el = $("#result-summary");
        const summary = d.summary;

        let html = `
            <div class="summary-card">
                <div class="num">${d.n_solutions}</div>
                <div class="lbl">パレート解の数</div>
            </div>
        `;

        // 各目的関数の最小値を表示
        for (const [name, stats] of Object.entries(summary.objectives)) {
            html += `
                <div class="summary-card">
                    <div class="num">${stats.min.toFixed(4)}</div>
                    <div class="lbl">${name} (最小)</div>
                </div>
            `;
        }

        el.innerHTML = html;
    }

    function renderParetoChart(d) {
        const objNames = d.obj_names;
        const objectives = d.objectives;

        if (objNames.length === 2) {
            // 2D scatter
            const trace = {
                x: objectives[objNames[0]],
                y: objectives[objNames[1]],
                mode: "markers",
                type: "scatter",
                marker: {
                    size: 6,
                    color: objectives[objNames[0]],
                    colorscale: "Viridis",
                    showscale: true,
                    colorbar: { title: objNames[0] },
                },
                text: objectives[objNames[0]].map(
                    (v, i) => `${objNames[0]}: ${v.toFixed(4)}<br>${objNames[1]}: ${objectives[objNames[1]][i].toFixed(4)}`
                ),
                hoverinfo: "text",
            };
            const layout = {
                title: "パレートフロント",
                xaxis: { title: objNames[0] },
                yaxis: { title: objNames[1] },
                margin: { l: 60, r: 30, t: 50, b: 50 },
                hovermode: "closest",
            };
            Plotly.newPlot("pareto-chart", [trace], layout, { responsive: true });
        } else if (objNames.length >= 3) {
            // 3D scatter (最初の3目的)
            const trace = {
                x: objectives[objNames[0]],
                y: objectives[objNames[1]],
                z: objectives[objNames[2]],
                mode: "markers",
                type: "scatter3d",
                marker: {
                    size: 4,
                    color: objectives[objNames[0]],
                    colorscale: "Viridis",
                    showscale: true,
                },
            };
            const layout = {
                title: "パレートフロント (3D)",
                scene: {
                    xaxis: { title: objNames[0] },
                    yaxis: { title: objNames[1] },
                    zaxis: { title: objNames[2] },
                },
                margin: { l: 0, r: 0, t: 50, b: 0 },
            };
            Plotly.newPlot("pareto-chart", [trace], layout, { responsive: true });
        }
    }

    function renderParallelChart(d) {
        const varNames = d.var_names;
        const variables = d.variables;
        const dimensions = varNames.map((name) => ({
            label: name,
            values: variables[name],
        }));

        // 色は最初の目的関数値で
        const objNames = d.obj_names;
        const colorValues = d.objectives[objNames[0]];

        const trace = {
            type: "parcoords",
            line: {
                color: colorValues,
                colorscale: "Viridis",
                showscale: true,
                colorbar: { title: objNames[0] },
            },
            dimensions: dimensions,
        };
        const layout = {
            title: "パラレル座標プロット（設計変数）",
            margin: { l: 60, r: 60, t: 50, b: 30 },
        };
        Plotly.newPlot("parallel-chart", [trace], layout, { responsive: true });
    }

    function renderTable(d) {
        const table = d.table;
        if (!table || table.length === 0) return;

        const cols = d.column_order || Object.keys(table[0]);

        // ヘッダー
        const head = $("#result-table-head");
        head.innerHTML = cols.map((c) => `<th data-col="${c}">${c}</th>`).join("");

        // ソートイベント
        head.querySelectorAll("th").forEach((th) => {
            th.addEventListener("click", () => {
                const col = th.dataset.col;
                if (S.sortCol === col) {
                    S.sortAsc = !S.sortAsc;
                } else {
                    S.sortCol = col;
                    S.sortAsc = true;
                }
                renderTableBody(table, cols);
            });
        });

        $("#table-count").textContent = `${table.length} 解`;
        renderTableBody(table, cols);
    }

    function renderTableBody(table, cols) {
        let rows = [...table];

        // ソート
        if (S.sortCol) {
            rows.sort((a, b) => {
                const va = a[S.sortCol], vb = b[S.sortCol];
                return S.sortAsc ? va - vb : vb - va;
            });
        }

        // 検索フィルタ
        const query = ($("#table-search").value || "").toLowerCase();
        if (query) {
            rows = rows.filter((row) =>
                cols.some((c) => String(row[c]).toLowerCase().includes(query))
            );
        }

        const body = $("#result-table-body");
        body.innerHTML = rows
            .map(
                (row, idx) =>
                    `<tr data-row-idx="${idx}">` + cols.map((c) => `<td>${row[c]}</td>`).join("") + "</tr>"
            )
            .join("");

        // 行クリックで予測シミュレータへ
        body.querySelectorAll("tr").forEach((tr) => {
            tr.addEventListener("click", () => {
                const idx = parseInt(tr.dataset.rowIdx);
                const row = rows[idx];
                if (row) populatePredictFromRow(row);
                body.querySelectorAll("tr").forEach((r) => r.classList.remove("selected-row"));
                tr.classList.add("selected-row");
            });
        });

        $("#table-count").textContent = `${rows.length} 解`;
    }

    function filterTable() {
        if (!S.resultData) return;
        const table = S.resultData.table;
        const cols = S.resultData.column_order || Object.keys(table[0]);
        renderTableBody(table, cols);
    }

    // ----- CSV ダウンロード ---------------------------------------------
    function downloadCSV() {
        if (!S.jobId) return;
        window.location.href = `/api/download/${S.jobId}`;
    }

    // ----- 予測シミュレータ -----------------------------------------------
    function renderPredictInputs(d) {
        const container = $("#predict-var-inputs");
        container.innerHTML = "";
        const varNames = d.var_names || [];
        // 最初の解の値をデフォルトとして使用
        const firstRow = (d.table && d.table[0]) || {};
        S.predictRefRow = firstRow;
        for (const name of varNames) {
            const val = firstRow[name] != null ? firstRow[name] : 0;
            const row = document.createElement("div");
            row.className = "predict-var-row";
            row.innerHTML = `
                <label title="${escapeHtml(name)}">${escapeHtml(name)}</label>
                <input type="number" step="any" data-var="${escapeHtml(name)}" value="${val}">
            `;
            container.appendChild(row);
        }

        // 予測ボタン（重複回避のため onclick を使用）
        $("#btn-predict").onclick = runPredict;

        // 予測結果をリセット
        $("#predict-output").innerHTML = '<p class="hint">変数値を入力して「予測」ボタンを押してください。</p>';
    }

    function populatePredictFromRow(row) {
        if (!S.resultData) return;
        S.predictRefRow = row;
        // querySelectorAll で全入力を走査（CSS セレクタの特殊文字問題を回避）
        const inputs = document.querySelectorAll("#predict-var-inputs input[data-var]");
        inputs.forEach((input) => {
            const vname = input.getAttribute("data-var");
            if (row[vname] != null) input.value = row[vname];
        });
        // タブを予測シミュレータに切り替え
        $$(".tab").forEach((t) => t.classList.remove("active"));
        $$(".tab-content").forEach((c) => c.classList.remove("active"));
        const predictTab = document.querySelector('.tab[data-tab="predict"]');
        if (predictTab) predictTab.classList.add("active");
        const predictContent = $("#tab-predict");
        if (predictContent) predictContent.classList.add("active");
    }

    async function runPredict() {
        if (!S.resultData) return;

        const values = {};
        const inputs = document.querySelectorAll("#predict-var-inputs input[data-var]");
        inputs.forEach((input) => {
            const name = input.getAttribute("data-var");
            values[name] = parseFloat(input.value) || 0;
        });

        const btn = $("#btn-predict");
        btn.disabled = true;
        btn.textContent = "計算中...";

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ job_id: S.jobId, config: S.config, values }),
            });
            if (!res.ok) {
                let msg = `サーバーエラー (${res.status})`;
                try { const d = await res.json(); if (d.error) msg = d.error; } catch (_) {}
                throw new Error(msg);
            }
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            renderPredictOutput(data);
        } catch (e) {
            $("#predict-output").innerHTML = `<div class="error-box"><strong>エラー</strong><p>${escapeHtml(e.message)}</p></div>`;
        } finally {
            btn.disabled = false;
            btn.textContent = "予測";
        }
    }

    function renderPredictOutput(data) {
        const out = $("#predict-output");
        const ref = S.predictRefRow || {};
        let html = "";

        // カスタムモード: 回帰予測値 vs 目標値
        if (data.predictions && Object.keys(data.predictions).length > 0) {
            html += '<table class="predict-table"><thead><tr>';
            html += '<th>特性</th><th>方向</th><th>回帰予測値</th><th>目標値</th><th>偏差</th><th>相対誤差</th>';
            html += '</tr></thead><tbody>';
            for (const [name, info] of Object.entries(data.predictions)) {
                const direction = info.direction || "target";
                const dirLabel = direction === "maximize" ? "最大化" : direction === "minimize" ? "最小化" : "目標値";

                if (direction === "target") {
                    const devPct = info.target !== 0 && info.target != null
                        ? Math.abs(info.abs_deviation / info.target * 100) : null;
                    let cls = "pr-good";
                    if (devPct !== null) {
                        if (devPct > 10) cls = "pr-bad";
                        else if (devPct > 3) cls = "pr-warn";
                    }
                    html += `<tr>`;
                    html += `<td class="pr-label">${escapeHtml(name)}</td>`;
                    html += `<td>${dirLabel}</td>`;
                    html += `<td class="${cls}">${info.predicted}</td>`;
                    html += `<td>${info.target}</td>`;
                    html += `<td>${info.abs_deviation}</td>`;
                    html += `<td class="${cls}">${devPct !== null ? devPct.toFixed(2) + "%" : "-"}</td>`;
                    html += `</tr>`;
                } else {
                    html += `<tr>`;
                    html += `<td class="pr-label">${escapeHtml(name)}</td>`;
                    html += `<td>${dirLabel}</td>`;
                    html += `<td>${info.predicted}</td>`;
                    html += `<td>-</td>`;
                    html += `<td>-</td>`;
                    html += `<td>-</td>`;
                    html += `</tr>`;
                }
            }
            html += '</tbody></table>';
        }

        // デモモード: 目的関数評価値 vs パレート解の値
        if (data.objectives && Object.keys(data.objectives).length > 0) {
            html += '<table class="predict-table"><thead><tr>';
            html += '<th>目的関数</th><th>計算値</th><th>パレート解の値</th><th>差分</th>';
            html += '</tr></thead><tbody>';
            for (const [name, val] of Object.entries(data.objectives)) {
                const refVal = ref[name];
                const diff = refVal != null ? Math.abs(val - refVal) : null;
                html += `<tr>`;
                html += `<td class="pr-label">${escapeHtml(name)}</td>`;
                html += `<td>${val}</td>`;
                html += `<td>${refVal != null ? refVal : "-"}</td>`;
                html += `<td>${diff != null ? diff.toFixed(6) : "-"}</td>`;
                html += `</tr>`;
            }
            html += '</tbody></table>';
        }

        out.innerHTML = html || '<p class="hint">予測結果がありません。</p>';
    }

    // ----- エラー表示 ---------------------------------------------------
    function showError(msg) {
        $("#error-area").classList.remove("hidden");
        $("#error-message").textContent = msg;
    }
})();
