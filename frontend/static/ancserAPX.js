/* ancserAPX — ancserTPX skin over the Alpaca factor backend.
   Backtest-only. Central chart overlays backtest + live + SPY curves. */
'use strict';

// ── State ───────────────────────────────────────────────────────────────────
let _ws = null;
let _config = null;
let _strategyPreset = null;  // active full-strategy preset name (e.g. "Claude #1")
let _activeBtab = 'holdings';
let _chart = null;
let _stratSeries = null;
let _liveSeries = null;
let _spySeries = null;
let _lastBacktestResult = null;
let _progressInterval = null;
let _connected = false;
let _secondaryFactors = [];  // factor names treated as secondary (二级)

// ── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    startClock();
    initWebSocket();
    loadConfig();
    initYearRange();
    initBottomDrag();
    initBottomTabs();
    loadDataStatus();
    // close conn dropdown when clicking outside
    document.addEventListener('click', (e) => {
        const wrap = document.querySelector('.conn-dropdown-wrap');
        if (wrap && !wrap.contains(e.target)) toggleConnDropdown(false);
    });
});

// ── Clock ──────────────────────────────────────────────────────────────────────
function startClock() {
    function tick() {
        const n = new Date();
        const p = (x) => String(x).padStart(2, '0');
        document.getElementById('clock').textContent =
            `${p(n.getUTCHours())}:${p(n.getUTCMinutes())}:${p(n.getUTCSeconds())} UTC`;
    }
    tick();
    setInterval(tick, 1000);
}

// ── WebSocket (log stream) ──────────────────────────────────────────────────────
function initWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    _ws = new WebSocket(`${proto}://${location.host}/ws`);
    _ws.onopen = () => { setInterval(() => { if (_ws.readyState === 1) _ws.send('ping'); }, 25000); };
    _ws.onclose = () => { setTimeout(initWebSocket, 5000); };
    _ws.onmessage = (ev) => {
        try {
            const m = JSON.parse(ev.data);
            if (m.type === 'log') appendLog(m.level, m.msg);
        } catch (e) {}
    };
}

// ── Connection status pill ───────────────────────────────────────────────────────
function setApiStatus(state) {
    const dot = document.getElementById('api-status');
    const txt = document.getElementById('api-status-text');
    const trg = document.getElementById('conn-trigger');
    const map = {
        connected:  { cls: 'ok',      label: 'CONNECTED' },
        ready:      { cls: 'loading', label: 'READY' },
        error:      { cls: 'err',     label: 'ERROR' },
        connecting: { cls: 'loading', label: 'CONNECTING' },
    };
    const s = map[state] || map.ready;
    dot.className = `status-dot ${s.cls}`;
    txt.textContent = s.label;
    trg.classList.toggle('connected', state === 'connected');
}

function toggleConnDropdown(force) {
    const panel = document.getElementById('conn-panel');
    const trg = document.getElementById('conn-trigger');
    const open = force !== undefined ? force : !panel.classList.contains('open');
    panel.classList.toggle('open', open);
    trg.classList.toggle('open', open);
}

// ── Logging ─────────────────────────────────────────────────────────────────────
function appendLog(level, msg) {
    const c = document.getElementById('log-container');
    if (!c) return;
    const clsMap = { info: 'log-info', success: 'log-success', warn: 'log-warn', error: 'log-error' };
    const d = document.createElement('div');
    d.className = `log-line ${clsMap[level] || ''}`;
    const n = new Date();
    const p = (x) => String(x).padStart(2, '0');
    d.innerHTML = `<span class="log-ts">${p(n.getHours())}:${p(n.getMinutes())}:${p(n.getSeconds())}</span> <span class="log-msg">${escHtml(msg)}</span>`;
    c.appendChild(d);
    c.scrollTop = c.scrollHeight;
    while (c.children.length > 500) c.removeChild(c.firstChild);
}

function escHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Config ───────────────────────────────────────────────────────────────────────
async function loadConfig() {
    try {
        const r = await fetch('/config');
        _config = await r.json();
        _secondaryFactors = _config.secondary_factors || [];
        buildFactorChecks(_config.factors || [], _config.factor_presets || {});
        buildPresetSelect(_config.factor_presets || {});
        setText('data-feed-info', _config.data_feed || 'IEX');
        const acct = (_config.accounts && _config.accounts[0]) || 'Main';
        setText('account-badge', acct);
        appendLog('info', `Config loaded — data feed: ${_config.data_feed || 'IEX'}, ${(_config.factors || []).length} factors`);
        setApiStatus('ready');
    } catch (e) {
        appendLog('error', `Config load failed: ${e.message}`);
        setApiStatus('error');
    } finally {
        // Decorate option labels with "?" help dots (fixed labels + MODEL
        // FACTORS header). Per-factor dots are added inside buildFactorChecks.
        decorateParamHelpDots();
    }
}

// ── Help dots + hover tooltips (ported from ancserTPX) ───────────────────────
// A small "?" dot is appended to each option's <label>. Hovering it shows a
// floating bilingual (中文 / English) tooltip explaining the option's usage.

function addHelpDot(label, tip) {
    if (!label || !tip || label.querySelector('.help-dot')) return;
    const dot = document.createElement('span');
    dot.className = 'help-dot';
    dot.textContent = '?';
    dot.setAttribute('data-tip', tip);
    dot.addEventListener('mouseenter', () => showHelpTooltip(dot));
    dot.addEventListener('mouseleave', hideHelpTooltip);
    // clicking the dot inside a <label> shouldn't toggle the checkbox
    dot.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); });
    label.appendChild(dot);
}

function getHelpTooltip() {
    let tip = document.getElementById('global-help-tooltip');
    if (!tip) {
        tip = document.createElement('div');
        tip.id = 'global-help-tooltip';
        tip.className = 'help-tooltip';
        document.body.appendChild(tip);
    }
    return tip;
}

function showHelpTooltip(dot) {
    const text = dot ? dot.getAttribute('data-tip') : '';
    if (!text) return;
    const tip = getHelpTooltip();
    tip.textContent = text;
    tip.style.visibility = 'hidden';
    tip.classList.add('open');
    const rect = dot.getBoundingClientRect();
    const pad = 10;
    const tipW = tip.offsetWidth;
    const tipH = tip.offsetHeight;
    const top = Math.max(pad, Math.min(
        rect.top + rect.height / 2 - tipH / 2,
        window.innerHeight - tipH - pad));
    const left = Math.max(pad, Math.min(
        rect.right + pad,
        window.innerWidth - tipW - pad));
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
    tip.style.visibility = 'visible';
}

function hideHelpTooltip() {
    const tip = document.getElementById('global-help-tooltip');
    if (tip) tip.classList.remove('open');
}

// Per-factor bilingual tips. Keyed by the factor's display name (see backend
// FACTOR_META). Used by buildFactorChecks() when rendering each factor row.
const FACTOR_HELP = {
    'Momentum':
        '一年期動量：過去 252 個交易日的總報酬，越高代表趨勢越強。買強勢股。\n' +
        '12-month total return (252d). Higher = stronger trend. Buys winners.',
    'Momentum 12-1':
        '經典橫斷面動量：12 個月報酬扣掉最近 1 個月（~21 日），避開短期反轉。\n' +
        '12-month return excluding the most recent month — classic cross-sectional momentum.',
    'Pullback 5d':
        '短期回檔：取近 5 日報酬的負值，跌得越多分數越高 → 在上升趨勢中逢低買入。\n' +
        'Negated 5-day return: bigger recent dip scores higher (buy-the-dip).',
    'Reversion':
        '均值回歸：以 RSI(14) 衡量超賣，RSI 越低分數越高，押注反彈。\n' +
        'Mean-reversion via RSI(14): lower RSI (oversold) scores higher.',
    'Skew':
        '已實現偏度（60 日）：偏好報酬分布右偏（偶有大漲）的股票。\n' +
        '60-day realized return skew — favors positively-skewed names.',
    'Microstructure':
        'Amihud 非流動性（20 日）：偏好流動性高、衝擊成本低的股票。\n' +
        '20-day Amihud illiquidity — prefers liquid, low-impact names.',
    'Alpha 101':
        'WorldQuant Alpha#6：open 與 volume 的 10 日負相關，捕捉量價背離。\n' +
        'WorldQuant Alpha#6: negative 10-day corr of open vs volume.',
    'Volatility':
        '特異波動率（20 日）：偏好低波動股票（低波動異象）。\n' +
        '20-day idiosyncratic volatility — prefers low-vol names (low-vol anomaly).',
    'Drift-Reversion':
        '漂移過濾的 RSI：只有在「非趨勢」盤整 regime 才啟用 RSI 回歸，趨勢中關閉。\n' +
        'RSI reversion that only activates outside drift (trending) regimes.',
    'Unicorn Edge':
        'Unicorn Edge：低價值 70% + 短期反轉 30%，且僅在多頭漂移 regime 啟動。\n' +
        'Low-price value (70%) + short-term reversal (30%), gated to drift regime.',
    'EMA200 Distance':
        '收盤價相對 200 日 EMA 的距離；越高代表越站在長期均線之上、越強勢。\n' +
        'Distance of close above/below the 200-day EMA. Higher = stronger.',
    'Rank Acceleration':
        '排名加速（二級）：過去 21 日綜合分排名是否往上爬升，捕捉「正在轉強」的股票。\n' +
        'Secondary: whether a name climbed in composite-score rank over ~21 days.',
    'Sector Rank':
        '板塊輪動（二級）：所屬 GICS 板塊 21 日總成交額成長率；資金流入的熱門板塊會抬升其中所有成分股。\n' +
        'Secondary: 21-day growth of the stock\u2019s GICS sector dollar-volume (hot-sector rotation).',
};

// Attach help dots to the fixed sidebar option labels. Factor-row dots are
// added separately inside buildFactorChecks(). Called once after the DOM and
// config have loaded.
function decorateParamHelpDots() {
    const labelTips = {
        'universe-select':
            '選股池。SPY + QQQ = S&P 500 與 NASDAQ 100 全部成分股（約 510 檔）；Custom = 自填代碼。\n' +
            '注意：此名單為「當前」成分，含倖存者偏差（已退市/被剔除股票未納入）。\n' +
            'Backtest universe. SPY+QQQ = current S&P 500 + NASDAQ 100 members (~510). Note: current membership only (survivorship bias).',
        'custom-symbols':
            '自訂股票代碼，用空格或逗號分隔。僅在 UNIVERSE = Custom 時生效。\n' +
            'Custom tickers (space/comma separated). Only used when UNIVERSE = Custom.',
        'capital-input':
            '回測起始資金（美元）。只影響回測曲線的金額尺度，不影響選股或實盤。\n' +
            'Starting capital (USD) for the backtest only. Scales the equity curve.',
        'year-start-slider':
            '回測年份範圍。拖動兩端設定起始與結束年（資料涵蓋約 2015–2026）。\n' +
            'Backtest year range — drag the two handles to set start/end year.',
        'preset-select':
            '策略預設。★ = 完整策略（含因子權重、槓桿、Top N、勝者鎖定）；其餘為純因子組合。\n' +
            '選「— custom —」則完全依照目前畫面上的手動設定。\n' +
            'Preset. ★ = full strategy (weights+leverage+TopN+lock); others = factor-only sets. "custom" uses the on-screen config.',
        'mwu-toggle':
            'MWU（Multiplicative Weights Update）動態權重。開啟後系統依各因子的歷史表現自動調整權重，手動權重欄位會變灰停用。\n' +
            'Dynamic factor weighting by past performance. When ON, manual weight dropdowns are grayed out.',
        'top-n-select':
            '每次調倉持有的股票數量 —— 取因子綜合分最高的前 N 檔等權持有。\n' +
            'Number of stocks held each rebalance — the top-N by composite factor score.',
        'leverage-slider':
            '槓桿倍數。1.0 = 滿倉不借錢；>1.0 = 借錢加倉（放大盈虧）；<1.0 = 保留部分現金。\n' +
            'Leverage. 1.0 = fully invested; >1 borrows (amplifies P&L); <1 holds cash.',
        'ema-kill-toggle':
            '200EMA 清倉開關。大盤（QQQ，缺則 SPY）跌破 200EMA 即全部清倉轉現金；之後每日檢查，直到站回 20EMA 才重新進場（重啟不受週五固定調倉限制）。\n' +
            '對高回撤策略有效降低 MaxDD；對純動量策略可能拖累報酬。\n' +
            '200EMA kill-switch: liquidate to cash when the market (QQQ/SPY) falls below its 200EMA; re-enter only after reclaiming the 20EMA (off the weekly cadence).',
    };
    Object.entries(labelTips).forEach(([id, tip]) => {
        const el = document.getElementById(id);
        const group = el ? el.closest('.form-group') : null;
        const label = group ? group.querySelector('label') : null;
        addHelpDot(label, tip);
    });
    // MODEL FACTORS section header (label has no input id of its own)
    const fLabel = [...document.querySelectorAll('.panel .form-group label')]
        .find(l => l.textContent.trim().startsWith('MODEL FACTORS'));
    addHelpDot(fLabel,
        '選股因子。勾選要使用的因子，右側下拉設定權重（— = 自動均權）。一級為主因子，二級為輔助微調。各因子的問號有詳細說明。\n' +
        'Stock-selection factors. Tick to enable, set weight on the right (— = auto/equal). Primary = core, Secondary = fine-tuning. Hover each factor\u2019s ? for details.');
}

// Build the weight dropdown for one factor row: "—" (auto/equal) then 0.0–1.0
// in steps of 0.1. The number is right-aligned (see .factor-weight CSS).
function _buildWeightSelect(factor) {
    const sel = document.createElement('select');
    sel.className = 'factor-weight';
    sel.dataset.factor = factor;
    const auto = document.createElement('option');
    auto.value = ''; auto.textContent = '—';
    sel.appendChild(auto);
    for (let i = 0; i <= 10; i++) {
        const v = (i / 10).toFixed(1);
        const o = document.createElement('option');
        o.value = v; o.textContent = v;
        sel.appendChild(o);
    }
    sel.onchange = (e) => { e.preventDefault(); markCustomPreset(); };
    // clicking the dropdown shouldn't toggle the parent <label>'s checkbox
    sel.onclick = (e) => e.stopPropagation();
    return sel;
}

function buildFactorChecks(factors, presets) {
    const primaryC = document.getElementById('factor-checks-primary');
    const secondaryC = document.getElementById('factor-checks-secondary');
    if (!primaryC || !secondaryC) return;
    primaryC.innerHTML = '';
    secondaryC.innerHTML = '';
    const defaultOn = presets['Baseline 70/30'] || presets['Balanced'] || factors.slice(0, 2);
    factors.forEach(f => {
        const isSecondary = _secondaryFactors.includes(f);
        const label = document.createElement('label');
        label.className = 'factor-chk';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = f;
        cb.checked = defaultOn.includes(f);
        cb.onchange = () => { markCustomPreset(); updateWeightDisabled(); };
        const name = document.createElement('span');
        name.className = 'factor-name';
        name.textContent = f;
        const wsel = _buildWeightSelect(f);
        label.appendChild(cb);
        label.appendChild(name);
        if (FACTOR_HELP[f]) addHelpDot(label, FACTOR_HELP[f]);
        label.appendChild(wsel);
        (isSecondary ? secondaryC : primaryC).appendChild(label);
    });
    updateWeightDisabled();
}

// Any factor-checkbox or live-affecting param change drops the preset to custom
// so Apply Live picks up the exact on-screen configuration.
function markCustomPreset() {
    const sel = document.getElementById('preset-select');
    if (sel) sel.value = '';
    _strategyPreset = null;
}

// Gray-out (disable) every weight dropdown while MWU is ON — MWU derives weights
// dynamically, so manual entry is meaningless. Also disable a factor's weight
// dropdown while that factor is unchecked.
function updateWeightDisabled() {
    const mwuOn = document.getElementById('mwu-hidden').value === '1';
    document.querySelectorAll('.factor-chk').forEach(row => {
        const cb = row.querySelector('input[type=checkbox]');
        const sel = row.querySelector('select.factor-weight');
        if (sel) sel.disabled = mwuOn || !cb || !cb.checked;
    });
}

// Read each row's weight dropdown into a {factor: weight} map. Returns null when
// no checked factor has an explicit weight (→ backend uses equal weighting).
// "—" (auto) factors mixed with explicit ones are sent as 0 (excluded).
function getFactorWeights() {
    const rows = [...document.querySelectorAll('.factor-chk')]
        .filter(r => r.querySelector('input[type=checkbox]')?.checked);
    const explicit = rows.filter(r => (r.querySelector('select.factor-weight')?.value || '') !== '');
    if (explicit.length === 0) return null;
    const out = {};
    rows.forEach(r => {
        const f = r.querySelector('input[type=checkbox]').value;
        const v = r.querySelector('select.factor-weight').value;
        out[f] = v === '' ? 0 : parseFloat(v);
    });
    return out;
}

// Apply a {factor: weight} map to the dropdowns (used by presets). Factors not
// in the map are reset to "—" (auto).
function setFactorWeights(weights) {
    document.querySelectorAll('.factor-chk').forEach(row => {
        const f = row.querySelector('input[type=checkbox]')?.value;
        const sel = row.querySelector('select.factor-weight');
        if (!sel) return;
        if (weights && weights[f] != null) {
            sel.value = parseFloat(weights[f]).toFixed(1);
        } else {
            sel.value = '';
        }
    });
}

function buildPresetSelect(presets) {
    const sel = document.getElementById('preset-select');
    if (!sel) return;
    const strat = _config.strategy_presets || {};
    // Full strategy presets first (sleeves + leverage + winner-lock)
    Object.keys(strat).forEach(name => {
        const o = document.createElement('option');
        o.value = name;
        o.textContent = '★ ' + name;
        sel.appendChild(o);
    });
    // Then plain factor presets
    const defaultPreset = presets['Baseline 70/30'] ? 'Baseline 70/30' : 'Balanced';
    Object.keys(presets).forEach(name => {
        const o = document.createElement('option');
        o.value = name;
        o.textContent = name;
        sel.appendChild(o);
    });
    // Default to the flagship strategy preset if present
    const firstStrat = Object.keys(strat)[0];
    const start = firstStrat || defaultPreset;
    sel.value = start;
    applyPreset(start);
}

function applyPreset(name) {
    if (!_config || !name) return;

    // ── Full strategy preset (sleeves + leverage + winner-lock) ──────────────
    const sp = (_config.strategy_presets || {})[name];
    if (sp) {
        _strategyPreset = name;
        // check the union of all sleeve factors so the UI reflects what runs
        const union = [];
        const mergedW = {};
        (sp.sleeves || []).forEach(sl => (sl.factors || []).forEach(f => {
            if (!union.includes(f)) union.push(f);
            // surface each sleeve's within-sleeve weights on the dropdowns
            if (sl.weights && sl.weights[f] != null) mergedW[f] = sl.weights[f];
        }));
        document.querySelectorAll('#factor-checks-primary input[type=checkbox], #factor-checks-secondary input[type=checkbox]').forEach(cb => {
            cb.checked = union.includes(cb.value);
        });
        setFactorWeights(mergedW);
        updateWeightDisabled();
        if (sp.leverage != null) {
            const lev = document.getElementById('leverage-slider');
            if (lev) { lev.value = sp.leverage; document.getElementById('leverage-val').textContent = parseFloat(sp.leverage).toFixed(1) + 'x'; }
        }
        if (sp.top_n != null) {
            const topSel = document.getElementById('top-n-select');
            if (topSel && [...topSel.options].some(o => o.value === String(sp.top_n))) topSel.value = String(sp.top_n);
        }
        if (sp.universe) {
            const uniSel = document.getElementById('universe-select');
            if (uniSel && [...uniSel.options].some(o => o.value === sp.universe)) { uniSel.value = sp.universe; onUniverseChange(); }
        }
        const sleeveStr = (sp.sleeves || []).map(s =>
            `${s.name} ${Math.round(s.alloc * 100)}%`).join(' / ');
        appendLog('info', `★ Strategy "${name}" — ${sp.label || ''}`);
        appendLog('info', `   leverage ${sp.leverage}x · top${sp.top_n} · sleeves: ${sleeveStr}`);
        return;
    }

    // ── Plain factor preset ──────────────────────────────────────────────────
    _strategyPreset = null;
    const factors = _config.factor_presets[name] || [];
    document.querySelectorAll('#factor-checks-primary input[type=checkbox], #factor-checks-secondary input[type=checkbox]').forEach(cb => {
        cb.checked = factors.includes(cb.value);
    });

    // Static factor weights for this preset → populate the weight dropdowns.
    const wp = (_config.factor_weight_presets || {})[name] || null;
    setFactorWeights(wp);
    updateWeightDisabled();

    // Preset-recommended defaults: top_n, universe, etc.
    const defaults = (_config.preset_defaults || {})[name];
    if (defaults) {
        if (defaults.top_n != null) {
            const topSel = document.getElementById('top-n-select');
            if (topSel) {
                const want = String(defaults.top_n);
                if ([...topSel.options].some(o => o.value === want)) topSel.value = want;
            }
        }
        if (defaults.universe) {
            const uniSel = document.getElementById('universe-select');
            if (uniSel && [...uniSel.options].some(o => o.value === defaults.universe)) {
                uniSel.value = defaults.universe;
                onUniverseChange();
            }
        }
    }
    appendLog('info', wp
        ? `Preset "${name}" — fixed weights ${Object.entries(wp).map(([k, v]) => `${k} ${Math.round(v * 100)}%`).join(' / ')}`
        : `Preset "${name}" applied.`);
}

// ── Year range (dual-handle slider) ────────────────────────────────────────────────
function initYearRange() {
    const cur = new Date().getFullYear();
    const a = document.getElementById('year-start-slider');
    const b = document.getElementById('year-end-slider');
    if (!a || !b) return;
    a.max = String(cur); b.max = String(cur);
    a.value = String(Math.min(2020, cur)); b.value = String(cur);
    const onInput = () => {
        let s = parseInt(a.value), e = parseInt(b.value);
        if (s > e) { // keep handles from crossing
            if (document.activeElement === a) { e = s; b.value = String(e); }
            else { s = e; a.value = String(s); }
        }
        updateYearRangeUI();
    };
    a.addEventListener('input', onInput);
    b.addEventListener('input', onInput);
    updateYearRangeUI();
}

function updateYearRangeUI() {
    const a = document.getElementById('year-start-slider');
    const b = document.getElementById('year-end-slider');
    const lbl = document.getElementById('year-range-val');
    const fill = document.getElementById('year-fill');
    if (!a || !b) return;
    const s = parseInt(a.value), e = parseInt(b.value);
    const lo = parseInt(a.min), hi = parseInt(a.max);
    const span = (hi - lo) || 1;
    if (lbl) lbl.textContent = `${s} — ${e}`;
    if (fill) {
        const left = ((s - lo) / span) * 100;
        const right = ((e - lo) / span) * 100;
        fill.style.left = left + '%';
        fill.style.width = Math.max(0, right - left) + '%';
    }
}

function getYearRange() {
    const s = parseInt(document.getElementById('year-start-slider').value);
    const e = parseInt(document.getElementById('year-end-slider').value);
    return { start_date: `${s}-01-01`, end_date: `${e}-12-31` };
}

function onUniverseChange() {
    const v = document.getElementById('universe-select').value;
    document.getElementById('custom-symbols-wrap').style.display = v === 'custom' ? '' : 'none';
}

function toggleField(btnId, hiddenId) {
    const btn = document.getElementById(btnId);
    const hidden = document.getElementById(hiddenId);
    const isOn = hidden.value === '1';
    hidden.value = isOn ? '0' : '1';
    btn.textContent = isOn ? 'OFF' : 'ON';
    btn.classList.toggle('on', !isOn);
    // MWU affects LIVE trading → drop to custom preset so Apply Live reflects it.
    markCustomPreset();
    // MWU toggle grays/ungrays the per-factor weight dropdowns.
    if (hiddenId === 'mwu-hidden') updateWeightDisabled();
}

// ── Bottom tabs / drag ──────────────────────────────────────────────────────────
function initBottomTabs() {
    document.querySelectorAll('.bottom-tab[data-btab]').forEach(tab => {
        tab.addEventListener('click', () => {
            const t = tab.dataset.btab;
            document.querySelectorAll('.bottom-tab').forEach(x => x.classList.remove('active'));
            tab.classList.add('active');
            document.querySelectorAll('[id^="btab-"]').forEach(p => p.classList.add('hidden'));
            document.getElementById(`btab-${t}`).classList.remove('hidden');
            _activeBtab = t;
        });
    });
}

function initBottomDrag() {
    const handle = document.getElementById('bottom-drag-handle');
    const bottom = document.getElementById('bottom-panel');
    const content = document.querySelector('.content');
    if (!handle || !bottom || !content) return;
    let startY = 0, startH = 0;
    handle.addEventListener('mousedown', e => {
        startY = e.clientY; startH = bottom.offsetHeight;
        document.addEventListener('mousemove', onDrag);
        document.addEventListener('mouseup', () => document.removeEventListener('mousemove', onDrag), { once: true });
    });
    function onDrag(e) {
        const newH = Math.max(80, Math.min(startH + (startY - e.clientY), content.offsetHeight - 100));
        bottom.style.height = newH + 'px';
        if (_chart) _chart.timeScale().fitContent();
    }
}

// ── Data status / connect / fetch ─────────────────────────────────────────────────
async function loadDataStatus() {
    try {
        const r = await fetch('/data/status');
        const d = await r.json();
        setText('data-coverage-pct', d.coverage_pct != null ? d.coverage_pct.toFixed(1) + '%' : '--');
        setText('data-coverage-detail', `${d.covered || 0}/${d.total || 0} · last ${d.last_update || 'never'}`);
    } catch (e) {
        setText('data-coverage-detail', 'unavailable');
    }
}

async function connectAPI() {
    const btn = document.getElementById('btn-connect');
    btn.disabled = true; btn.textContent = 'CONNECTING…';
    setApiStatus('connecting');
    appendLog('info', 'Verifying Alpaca connection via .env credentials…');
    try {
        const r = await fetch('/live/status?account=Main');
        const d = await r.json();
        if (d.error) {
            appendLog('error', `Connection failed: ${d.error}`);
            setApiStatus('error');
        } else {
            const acct = d.account || {};
            const num = acct.account_number ? '••' + String(acct.account_number).slice(-4) : 'paper';
            setText('conn-account-info', num);
            const eq = parseFloat(acct.equity || 0);
            appendLog('success', `Connected. Account ${num} · equity $${eq.toLocaleString()}`);
            setApiStatus('connected');
            _connected = true;
            // get + store data (incremental — only fills what's missing)
            triggerFetch('incremental');
        }
    } catch (e) {
        appendLog('error', `Connection error: ${e.message}`);
        setApiStatus('error');
    } finally {
        btn.disabled = false; btn.textContent = 'CONNECT & SYNC';
    }
}

async function triggerFetch(mode) {
    const btn = mode === 'bulk' ? document.getElementById('btn-fetch-full') : null;
    if (btn) { btn.disabled = true; btn.textContent = 'FETCHING…'; }
    appendLog('info', `${mode === 'bulk' ? 'Full' : 'Incremental'} data fetch starting (stores what is missing)…`);
    switchBtab('log');
    try {
        const body = { mode, account: 'Main' };
        if (mode === 'bulk') body.start_date = '2015-01-01';
        const r = await fetch('/data/fetch', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const res = await r.json();
        appendLog('success', `Fetch complete: ${JSON.stringify(res)}`);
        await loadDataStatus();
    } catch (e) {
        appendLog('error', `Fetch failed: ${e.message}`);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'FETCH FULL HISTORY (10Y)'; }
    }
}

// ── Backtest ──────────────────────────────────────────────────────────────────────
function getActiveFactors() {
    const out = [];
    document.querySelectorAll('#factor-checks-primary input[type=checkbox]:checked, #factor-checks-secondary input[type=checkbox]:checked').forEach(cb => out.push(cb.value));
    return out;
}

function getCustomSymbols() {
    const raw = document.getElementById('custom-symbols').value.trim();
    if (!raw) return null;
    return raw.split(/[\s,]+/).map(s => s.trim().toUpperCase()).filter(Boolean);
}

async function runBacktest() {
    const factors = getActiveFactors();
    if (factors.length === 0) {
        appendLog('warn', 'Select at least one factor before running backtest.');
        return;
    }
    const universe = document.getElementById('universe-select').value;
    const yr = getYearRange();
    const body = {
        universe,
        symbols: universe === 'custom' ? getCustomSymbols() : null,
        start_date: yr.start_date,
        end_date: yr.end_date,
        active_factors: factors,
        capital: parseFloat(document.getElementById('capital-input').value) || 100000,
        leverage: parseFloat(document.getElementById('leverage-slider').value) || 1.0,
        use_mwu: document.getElementById('mwu-hidden').value === '1',
        top_n: parseInt(document.getElementById('top-n-select').value) || 30,
        factor_weights: getFactorWeights(),
        strategy_preset: _strategyPreset || null,
        ema_kill_switch: document.getElementById('ema-kill-hidden').value === '1',
    };

    const btn = document.getElementById('btn-run-backtest');
    btn.disabled = true; btn.textContent = 'RUNNING…';
    document.getElementById('backtest-progress-wrap').style.display = '';
    switchBtab('log');
    startProgressAnimation();
    appendLog('info', `Backtest starting: ${factors.join(', ')}`);

    try {
        const r = await fetch('/backtest/run', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const result = await r.json();
        stopProgressAnimation(!result.error);
        if (result.error) {
            appendLog('error', `Backtest error: ${result.error}`);
        } else {
            _lastBacktestResult = result;
            renderMetrics(result.metrics);
            renderComparisonChart(result);
            renderHoldings(result.holdings || []);
            renderFactorWeights(result.factor_weights || [], result.params?.factors || []);
            appendLog('success', `Backtest complete — CAGR ${result.metrics?.cagr_pct?.toFixed(1)}% · Sharpe ${result.metrics?.sharpe?.toFixed(2)}`);
        }
    } catch (e) {
        stopProgressAnimation(false);
        appendLog('error', `Backtest request failed: ${e.message}`);
    } finally {
        btn.disabled = false; btn.textContent = 'EXECUTE BACKTEST';
    }
}

async function applyLive() {
    const factors = getActiveFactors();
    const body = {
        account: (_config && _config.accounts && _config.accounts[0]) || 'Main',
        strategy_preset: _strategyPreset || null,
        active_factors: factors,
        // universe is a backtest-only knob — live always trades the full SPY+QQQ set
        universe: 'spy_qqq',
        leverage: parseFloat(document.getElementById('leverage-slider').value) || 1.0,
        top_n: parseInt(document.getElementById('top-n-select').value) || 20,
        factor_weights: getFactorWeights(),
        use_mwu: document.getElementById('mwu-hidden').value === '1',
        ema_kill_switch: document.getElementById('ema-kill-hidden').value === '1',
    };

    const label = _strategyPreset ? `strategy "${_strategyPreset}"` : `custom factors [${factors.join(', ')}]`;
    if (!confirm(`Apply ${label} to LIVE trading?\n\nIt will be executed on the next daily rebalance.`)) return;

    const btn = document.getElementById('btn-apply-live');
    if (btn) { btn.disabled = true; btn.textContent = 'APPLYING…'; }
    switchBtab('log');
    appendLog('warn', `Applying ${label} to live config…`);
    try {
        const r = await fetch('/live/apply', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const res = await r.json();
        if (res.error) {
            appendLog('error', `Apply live failed: ${res.error}`);
        } else {
            appendLog('success', `✓ Live strategy updated: ${res.description}`);
            appendLog('info', '→ Takes effect on the NEXT daily rebalance run.');
        }
    } catch (e) {
        appendLog('error', `Apply live request failed: ${e.message}`);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'APPLY LIVE'; }
    }
}

function startProgressAnimation() {
    let pct = 0;
    const bar = document.getElementById('backtest-progress-bar');
    const txt = document.getElementById('backtest-progress-text');
    _progressInterval = setInterval(() => {
        if (pct < 60) pct += 2.5; else if (pct < 85) pct += 0.5;
        bar.style.width = pct + '%';
        txt.textContent = pct < 85 ? 'computing…' : 'finalizing…';
    }, 300);
}

function stopProgressAnimation(success) {
    if (_progressInterval) { clearInterval(_progressInterval); _progressInterval = null; }
    const bar = document.getElementById('backtest-progress-bar');
    const txt = document.getElementById('backtest-progress-text');
    bar.style.width = '100%';
    bar.style.background = success ? 'var(--green)' : 'var(--red)';
    txt.textContent = success ? 'done' : 'failed';
    setTimeout(() => {
        document.getElementById('backtest-progress-wrap').style.display = 'none';
        bar.style.width = '0%'; bar.style.background = 'var(--green)';
    }, 1500);
}

// ── Metrics ─────────────────────────────────────────────────────────────────────
function renderMetrics(m) {
    if (!m) return;
    document.getElementById('metrics-panel').style.display = '';
    const grid = document.getElementById('metrics-grid');

    // Liquidation (爆倉) drawdown for the leverage used: Reg-T 25% maintenance
    // margin → equity-DD threshold = (1 − 0.25·L)/(1 − 0.25). If MaxDD breaches
    // it, flag with a "!".
    const lev = (_lastBacktestResult && _lastBacktestResult.params && _lastBacktestResult.params.leverage) || 1.0;
    const liqDD = (1 - 0.25 * lev) / 0.75;            // fraction, e.g. L=1.5 → 0.833
    const maxddFrac = Math.abs(m.max_dd_pct || 0) / 100;
    const breached = maxddFrac >= liqDD;
    const maxddStr = fmtNum(m.max_dd_pct, 1) + '%' + (breached ? ' !' : '');

    const rows = [
        { l: 'FINAL EQUITY', v: '$' + fmtNum(m.final_equity, 0), c: 'pos' },
        { l: 'TOTAL RETURN', v: fmtNum(m.total_return_pct, 1) + '%', c: m.total_return_pct >= 0 ? 'pos' : 'neg' },
        { l: 'CAGR', v: fmtNum(m.cagr_pct, 1) + '%', c: m.cagr_pct >= 0 ? 'pos' : 'neg' },
        { l: 'SHARPE', v: fmtNum(m.sharpe, 2), c: m.sharpe >= 1 ? 'pos' : '' },
        { l: 'CALMAR', v: fmtNum(m.calmar, 2), c: '' },
        { l: 'MAX DD', v: maxddStr, c: 'neg' },
        { l: 'WIN RATE', v: fmtNum(m.win_rate_pct, 1) + '%', c: '' },
        { l: 'DAYS', v: (m.total_days || 0).toLocaleString(), c: '' },
    ];
    if (breached) appendLog('warn', `MaxDD ${fmtNum(m.max_dd_pct,1)}% breaches the ${(liqDD*100).toFixed(1)}% liquidation drawdown at ${lev}x leverage (爆倉風險).`);
    grid.innerHTML = rows.map(r =>
        `<div class="metric-card"><div class="label">${r.l}</div><div class="value ${r.c}">${r.v}</div></div>`
    ).join('');
}

function fmtNum(v, dec) {
    if (v == null || isNaN(v)) return '--';
    return Number(v).toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

// ── Comparison chart: backtest + live + SPY ───────────────────────────────────────
function renderComparisonChart(result) {
    const container = document.getElementById('chart-container');
    document.getElementById('chart-placeholder').style.display = 'none';
    document.getElementById('chart-legend').style.display = 'flex';

    if (_chart) { _chart.remove(); _chart = null; _stratSeries = _liveSeries = _spySeries = null; }

    _chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: { background: { color: '#08090d' }, textColor: '#8a8f9e' },
        grid: { vertLines: { color: '#1c1e28' }, horzLines: { color: '#1c1e28' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#1c1e28' },
        timeScale: { borderColor: '#1c1e28', timeVisible: false, secondsVisible: false },
        localization: {
            dateFormat: 'MM.dd.yyyy',
            timeFormatter: (t) => {
                // t is a business-day string ("YYYY-MM-DD") for daily data
                if (typeof t === 'string') {
                    const [y, m, d] = t.split('-');
                    return `${m}.${d}.${y}`;
                }
                const dt = new Date((typeof t === 'number' ? t * 1000 : t));
                const mm = String(dt.getUTCMonth() + 1).padStart(2, '0');
                const dd = String(dt.getUTCDate()).padStart(2, '0');
                return `${mm}.${dd}.${dt.getUTCFullYear()}`;
            },
        },
    });

    const benchLabel = result.benchmark || 'QQQ';
    _stratSeries = _chart.addLineSeries({ color: '#00e5a0', lineWidth: 2, title: 'Strategy' });
    _spySeries = _chart.addLineSeries({ color: '#ffa726', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, title: benchLabel });
    _liveSeries = _chart.addLineSeries({ color: '#64dcff', lineWidth: 2, title: 'Live' });

    if (result.equity_curve?.length)
        _stratSeries.setData(result.equity_curve.map(d => ({ time: d.date, value: d.value })));
    const benchCurve = result.benchmark_curve || result.spy_curve;
    if (benchCurve?.length)
        _spySeries.setData(benchCurve.map(d => ({ time: d.date, value: d.value })));

    _chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
        if (_chart) _chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    });
    ro.observe(container);

    // Best-effort live account overlay (read-only portfolio history)
    overlayLiveCurve();
}

async function overlayLiveCurve() {
    try {
        const r = await fetch('/live/equity?account=Main&period=all');
        const d = await r.json();
        const curve = d.equity_curve || [];
        if (!curve.length || !_liveSeries) {
            if (_liveSeries) appendLog('info', 'No live account history to overlay.');
            return;
        }

        // ── Time + money alignment ───────────────────────────────────────────
        // The live account starts whenever it was funded (e.g. 2026) with its
        // own real dollar balance. To compare it against a backtest that began
        // earlier (e.g. 2022) we re-base the live curve: it starts at the value
        // the *strategy* (backtest) curve had on the live start date, then
        // tracks the live account's actual % returns from there. This way both
        // lines share the same money scale and connect in time.
        const strat = (_lastBacktestResult && _lastBacktestResult.equity_curve) || [];
        const liveStart = curve[0].date;
        let anchorVal = null;
        if (strat.length) {
            // last strategy point on/before the live start date
            for (const p of strat) {
                if (p.date <= liveStart) anchorVal = p.value;
                else break;
            }
            // live started after backtest ended → continue from final equity
            if (anchorVal == null) anchorVal = strat[strat.length - 1].value;
        }

        const base = curve[0].value || 0;
        const scale = (anchorVal != null && base > 0) ? (anchorVal / base) : 1;
        _liveSeries.setData(curve.map(p => ({ time: p.date, value: Math.round(p.value * scale * 100) / 100 })));
        if (scale !== 1) {
            appendLog('info', `Live overlay: ${curve.length} days, re-based to strategy equity on ${liveStart} (×${scale.toFixed(3)}).`);
        } else {
            appendLog('info', `Live account overlay: ${curve.length} days.`);
        }
    } catch (e) {
        // live overlay is optional; ignore failures
    }
}

// ── Tables ──────────────────────────────────────────────────────────────────────
function renderHoldings(rows) {
    const tbody = document.getElementById('holdings-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    const data = rows.slice(-60);
    const parse = (s) => String(s || '').split(/[\s,]+/).map(x => x.trim().toUpperCase()).filter(Boolean);
    const symsByRow = data.map(r => parse(r.long));

    // Current holding tenure = number of consecutive most-recent rows a symbol
    // appears in (counting backwards from the latest row, stops at first gap).
    const tenure = {};
    const allSyms = new Set();
    symsByRow.forEach(arr => arr.forEach(s => allSyms.add(s)));
    allSyms.forEach(sym => {
        let streak = 0;
        for (let i = symsByRow.length - 1; i >= 0; i--) {
            if (symsByRow[i].includes(sym)) streak++;
            else break;
        }
        tenure[sym] = streak;
    });
    // Global ordering: longest tenure first, ties alphabetical → columns align.
    const rank = (sym) => -tenure[sym] * 1000 + sym.charCodeAt(0);

    data.reverse().forEach((r, idx) => {
        const ordered = parse(r.long).sort((a, b) => rank(a) - rank(b));
        const chips = ordered.map(s => {
            // brighten names held the entire visible window
            const long = tenure[s] >= symsByRow.length;
            return `<span class="hold-chip"${long ? ' style="color:var(--amber)"' : ''}>${s}</span>`;
        }).join('');
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${r.date}</td><td>${chips || '--'}</td>`;
        tbody.appendChild(tr);
    });
}

function renderFactorWeights(rows, factors) {
    const thead = document.getElementById('weights-thead-row');
    const tbody = document.getElementById('weights-tbody');
    if (!thead || !tbody) return;
    thead.innerHTML = '<th>DATE</th>' + factors.map(f => `<th>${f}</th>`).join('');
    tbody.innerHTML = '';
    rows.slice(-60).reverse().forEach(r => {
        let html = `<td>${r.date}</td>`;
        factors.forEach(f => { html += `<td>${r[f] != null ? (r[f] * 100).toFixed(1) + '%' : '--'}</td>`; });
        const tr = document.createElement('tr');
        tr.innerHTML = html;
        tbody.appendChild(tr);
    });
}

// ── Helpers ─────────────────────────────────────────────────────────────────────
function setText(id, t) { const el = document.getElementById(id); if (el) el.textContent = t; }

function switchBtab(tab) {
    const t = document.querySelector(`.bottom-tab[data-btab="${tab}"]`);
    if (t) t.click();
}
