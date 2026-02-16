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
        uploadedModels: [],     // [{path, filename, features}]
        jobId: null,
        pollTimer: null,
        resultData: null,
        sortCol: null,
        sortAsc: true,
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
                <td><input type="number" class="tgt-val" value="0" step="any"></td>
                <td><button class="btn-icon btn-remove-tgt" title="削除">&times;</button></td>
            `;
            $("#target-table-body").appendChild(row);
            bindRemoveButtons();
        });

        // モデルアップロード
        $("#model-file-input").addEventListener("change", handleModelUpload);

        bindRemoveButtons();
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
                renderUploadedModels();
            } catch (err) {
                alert("アップロードに失敗しました: " + err.message);
            }
        }
        e.target.value = "";
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
                <span>${escapeHtml(m.filename)} (${m.features.length} 特徴量)</span>
                <button class="btn-icon" data-idx="${i}" title="削除">&times;</button>
            </div>
        `
            )
            .join("");
        // イベントデリゲーション（inline onclick を回避）
        el.querySelectorAll(".btn-icon").forEach((btn) => {
            btn.addEventListener("click", () => {
                S.uploadedModels.splice(parseInt(btn.dataset.idx), 1);
                renderUploadedModels();
            });
        });
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
        $("#btn-to-step2").addEventListener("click", () => goToStep(2));
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
        $$("#target-table-body tr").forEach((row) => {
            const name = row.querySelector(".tgt-name").value.trim();
            const val = parseFloat(row.querySelector(".tgt-val").value);
            if (name) targets[name] = val;
        });

        const modelPaths = {};
        S.uploadedModels.forEach((m) => {
            // filenameから特性名を推定（例: ols_特性A.joblib → 特性A）
            let key = m.filename.replace(/^ols_/, "").replace(/\.joblib$/, "");
            modelPaths[key] = m.path;
        });

        const aggMode = document.querySelector('input[name="agg-mode"]:checked').value;
        const devMode = document.querySelector('input[name="dev-mode"]:checked').value;

        return {
            mode: "custom",
            variables,
            model_paths: modelPaths,
            targets,
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
        const varNames = S.resultData.var_names || [];
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
        if (!S.jobId || !S.resultData) return;

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
                body: JSON.stringify({ job_id: S.jobId, values }),
            });
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
        let html = "";

        // 目的関数値（デモモード）
        if (data.objectives) {
            html += '<div class="predict-result-card"><div class="pr-name">目的関数値</div>';
            for (const [name, val] of Object.entries(data.objectives)) {
                html += `<div class="pr-row"><span class="pr-label">${escapeHtml(name)}</span><span class="pr-val">${val}</span></div>`;
            }
            html += "</div>";
        }

        // 予測値（カスタムモード）
        if (data.predictions) {
            for (const [name, info] of Object.entries(data.predictions)) {
                const devPct = info.target !== 0 ? Math.abs(info.abs_deviation / info.target * 100) : null;
                let colorClass = "pr-good";
                if (devPct !== null) {
                    if (devPct > 10) colorClass = "pr-bad";
                    else if (devPct > 3) colorClass = "pr-warn";
                }
                html += `<div class="predict-result-card">`;
                html += `<div class="pr-name">${escapeHtml(name)}</div>`;
                html += `<div class="pr-row"><span class="pr-label">目標値</span><span class="pr-val">${info.target}</span></div>`;
                html += `<div class="pr-row"><span class="pr-label">予測値</span><span class="pr-val ${colorClass}">${info.predicted}</span></div>`;
                html += `<div class="pr-row"><span class="pr-label">絶対偏差</span><span class="pr-val">${info.abs_deviation}</span></div>`;
                html += `<div class="pr-row"><span class="pr-label">偏差指標</span><span class="pr-val">${info.deviation}</span></div>`;
                if (devPct !== null) {
                    html += `<div class="pr-row"><span class="pr-label">相対誤差</span><span class="pr-val ${colorClass}">${devPct.toFixed(2)}%</span></div>`;
                }
                html += `</div>`;
            }
        }

        out.innerHTML = html || '<p class="hint">予測結果がありません。</p>';
    }

    // ----- エラー表示 ---------------------------------------------------
    function showError(msg) {
        $("#error-area").classList.remove("hidden");
        $("#error-message").textContent = msg;
    }
})();
