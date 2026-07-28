/* ancserAPX — ancserTPX skin over the Alpaca factor backend.
   Backtest-only. Central chart overlays backtest + live + SPY curves. */
'use strict';

// ── State ───────────────────────────────────────────────────────────────────
let _ws = null;
let _config = null;
let _strategyPreset = null;  // active full-strategy preset name (e.g. "Claude #1")
let _selectedPresetLabel = null; // includes factor-only presets for local history labels
let _activeBtab = 'holdings';
let _chart = null;
let _stratSeries = null;
let _liveSeries = null;
let _spySeries = null;
let _lastBacktestResult = null;
let _progressInterval = null;
let _connected = false;
let _selectedAccount = 'Main';
let _chartViewMode = '1y';
let _activeView = 'backtest';
let _currentLiveSnapshot = null;
let _applyingPreset = false;
let _historyDb = null;
let _historyRecords = [];

const HISTORY_DB_NAME = 'ancserAPXResults';
const HISTORY_STORE = 'results';
const HISTORY_DB_VERSION = 1;

// ── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    initTheme();
    startClock();
    initWebSocket();
    loadConfig();
    initYearRange();
    initBottomDrag();
    initBottomTabs();
    initWorkspaceTabs();
    initSettingsDirtyTracking();
    initApxGlassStandard();
    initHistoryStore().then(refreshHistoryList);
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

// ── APX Liquid Glass standard helpers ─────────────────────────────────────
// These helpers standardize native controls without changing their submitted
// values. The WebGL liquid-glass demo remains the shader calibration source.
function syncApxGlassRange(input) {
    if (!input) return;
    const min = Number(input.min || 0);
    const max = Number(input.max || 100);
    const value = Number(input.value || min);
    const pct = max > min
        ? ((value - min) / (max - min)) * 100
        : 0;
    input.style.setProperty(
        '--apx-range-progress',
        `${Math.max(0, Math.min(100, pct)).toFixed(2)}%`
    );
}

function syncApxFactorSwitches(root = document) {
    root.querySelectorAll('.factor-chk').forEach(row => {
        const checked = row.querySelector('input[type=checkbox]')?.checked;
        row.classList.toggle('is-on', Boolean(checked));
    });
}

// ── Light / dark mode ─────────────────────────────────────────────────────
// The palette lives in CSS custom properties, so switching themes is a single
// attribute flip; the chart and the WebGL glass renderer are told to re-read.
const APX_THEME_KEY = 'ancserAPXTheme';

function chartThemeOptions() {
    const light = document.documentElement.dataset.theme === 'light';
    const border = light ? '#d3d5d0' : '#1c1e28';
    return {
        layout: {
            background: { color: light ? '#f3f3ef' : '#08090d' },
            textColor: light ? '#5b6474' : '#8a8f9e',
        },
        grid: { vertLines: { color: border }, horzLines: { color: border } },
        rightPriceScale: { borderColor: border },
        timeScale: { borderColor: border },
    };
}

function setTheme(theme, opts = {}) {
    const light = theme === 'light';
    document.documentElement.dataset.theme = light ? 'light' : 'dark';
    const toggle = document.getElementById('theme-toggle');
    if (toggle && toggle.checked !== light) toggle.checked = light;
    const icon = document.getElementById('theme-icon');
    if (icon) icon.textContent = light ? '☀' : '☾';
    if (_chart) _chart.applyOptions(chartThemeOptions());
    if (window.ApxGlass) window.ApxGlass.invalidate(true);
    if (!opts.silent) {
        try { localStorage.setItem(APX_THEME_KEY, light ? 'light' : 'dark'); } catch (e) {}
    }
}

function initTheme() {
    let stored = null;
    try { stored = localStorage.getItem(APX_THEME_KEY); } catch (e) {}
    setTheme(stored === 'light' ? 'light' : 'dark', { silent: true });
}

function initApxGlassStandard(root = document) {
    root.querySelectorAll('input[type=range].apx-glass-range').forEach(input => {
        syncApxGlassRange(input);
        if (input.dataset.apxGlassBound === '1') return;
        input.dataset.apxGlassBound = '1';
        input.addEventListener('input', () => syncApxGlassRange(input));
        input.addEventListener('change', () => syncApxGlassRange(input));
    });
    syncApxFactorSwitches(root);
    // apx-liquid-glass.js mounts the WebGL surfaces; it is a no-op without it.
    if (window.ApxGlass) window.ApxGlass.refresh();
}

function getAccountDetail(account) {
    const details = (_config && _config.account_details) || [];
    return details.find(a => a.name === account) || null;
}

function accountEnvKey(account) {
    const detail = getAccountDetail(account);
    if (detail && detail.key_env) return detail.key_env;
    return account === 'Main' ? 'APCA_API_KEY_ID' : `APCA_API_KEY_ID_${String(account).toUpperCase()}`;
}

function getSelectedAccount() {
    const sel = document.getElementById('live-account-select');
    return (sel && sel.value) || _selectedAccount || 'Main';
}

function getSelectedAccountMode(account = null) {
    const detail = getAccountDetail(account || getSelectedAccount());
    return (detail && detail.mode) || (detail && detail.paper === false ? 'live' : 'paper');
}

function setSelectedAccount(account, opts = {}) {
    const previous = _selectedAccount;
    _selectedAccount = account || 'Main';
    const mode = getSelectedAccountMode(_selectedAccount);
    const keyEnv = accountEnvKey(_selectedAccount);
    const sel = document.getElementById('live-account-select');
    if (sel && sel.value !== _selectedAccount) sel.value = _selectedAccount;
    const badge = document.getElementById('account-badge');
    if (badge) {
        badge.textContent = `${_selectedAccount} ${mode.toUpperCase()}`;
        badge.classList.toggle('funded', mode === 'live');
        badge.classList.toggle('practice', mode !== 'live');
    }
    setText('conn-account-info', `${_selectedAccount} (${mode.toUpperCase()})`);
    setText('account-env-info', keyEnv);
    if (!opts.silent && previous !== _selectedAccount) {
        _connected = false;
        setApiStatus('ready');
        appendLog('info', `Selected live account: ${_selectedAccount} (${mode}, ${keyEnv})`);
        if (_liveSeries) {
            _liveSeries.setData([]);
            overlayLiveCurve();
        }
        if (_activeView === 'live') loadLiveDashboard({ persist: true });
    }
}

function buildAccountSelect(accounts) {
    const sel = document.getElementById('live-account-select');
    const list = (accounts && accounts.length) ? accounts : ['Main'];
    const current = list.includes(_selectedAccount) ? _selectedAccount : list[0];
    if (sel) {
        sel.innerHTML = '';
        list.forEach(account => {
            const detail = getAccountDetail(account);
            const mode = ((detail && detail.mode) || (detail && detail.paper === false ? 'live' : 'paper')).toUpperCase();
            const o = document.createElement('option');
            o.value = account;
            o.textContent = `${account} - ${mode} - ${accountEnvKey(account)}`;
            sel.appendChild(o);
        });
    }
    setSelectedAccount(current, { silent: true });
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
        buildModelSelect(_config.models || []);
        buildFactorChecks(_config.factors || [], _config.factor_presets || {});
        buildPresetSelect(_config.factor_presets || {});
        buildAccountSelect(_config.accounts || ['Main']);
        setText('data-feed-info', _config.data_feed || 'IEX');
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
    'Pullback 5d':
        '短期回檔：取近 5 日報酬的負值，跌得越多分數越高 → 在上升趨勢中逢低買入。\n' +
        'Negated 5-day return: bigger recent dip scores higher (buy-the-dip).',
    'Reversion':
        '均值回歸：以 RSI(14) 衡量超賣，RSI 越低分數越高，押注反彈。\n' +
        'Mean-reversion via RSI(14): lower RSI (oversold) scores higher.',
    'Microstructure':
        'Amihud 非流動性（20 日）：偏好流動性高、衝擊成本低的股票。\n' +
        '20-day Amihud illiquidity — prefers liquid, low-impact names.',
    'Volatility':
        '特異波動率（20 日）：偏好低波動股票（低波動異象）。\n' +
        '20-day idiosyncratic volatility — prefers low-vol names (low-vol anomaly).',
    'EMA200 Distance':
        '收盤價相對 200 日 EMA 的距離；越高代表越站在長期均線之上、越強勢。\n' +
        'Distance of close above/below the 200-day EMA. Higher = stronger.',
    'MA VWAP Entry':
        'Daily MA/VWAP entry: SMA20 > SMA50 > SMA100, close above daily VWAP, close near SMA20, RSI not overbought. Intended to find trend pullback entries.',
};

// Attach help dots to the fixed sidebar option labels. Factor-row dots are
// added separately inside buildFactorChecks(). Called once after the DOM and
// config have loaded.
function decorateParamHelpDots() {
    const labelTips = {
        'universe-select':
            '選股池。SPY + QQQ = S&P 500 與 NASDAQ 100 名單（約 510 檔）；Custom = 自填代碼。\n' +
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
            '修改任何設定後會顯示 UNSAVED MODIFIED SET，表示目前畫面已不再等同原預設。\n' +
            'Preset. ★ = full strategy. Any edited setting changes this field to UNSAVED MODIFIED SET.',
        'model-select':
            '選股／打分模型。Factor Composite 使用下方因子及權重；不使用因子的模型只會在後端明確宣告可用時出現。\n' +
            'Stock scoring model. Factor Composite uses the factor list below. Non-factor models appear only when advertised by the backend.',
        'live-account-select':
            'Live account target for APPLY LIVE, CONNECT, data sync, and the live overlay.\n' +
            'Options are loaded from .env keys: Main uses APCA_API_KEY_ID; extra accounts use APCA_API_KEY_ID_<NAME>. Paper/live mode uses APCA_PAPER or APCA_PAPER_<NAME>.',
        'mwu-toggle':
            'MWU（Multiplicative Weights Update）動態權重。開啟後系統依各因子的歷史表現自動調整權重，手動權重欄位會變灰停用。\n' +
            'Dynamic factor weighting by past performance. When ON, manual weight dropdowns are grayed out.',
        'top-n-select':
            '每次調倉持有的股票數量 —— 取因子綜合分最高的前 N 檔等權持有。\n' +
            'Number of stocks held each rebalance — the top-N by composite factor score.',
        'holding-period-select':
            'Holding period between rebalances. 1D=1, 1W=5, 1M=21, 2M=42, 4M=84 trading days. Backtest and Apply Live both use this value.',
        'leverage-slider':
            '槓桿倍數。1.0 = 滿倉不借錢；>1.0 = 借錢加倉（放大盈虧）；<1.0 = 保留部分現金。\n' +
            'Leverage. 1.0 = fully invested; >1 borrows (amplifies P&L); <1 holds cash.',
        'commission-bps-input':
            'Broker commission per traded notional, in basis points. Alpaca US stock commission defaults to 0 bps.',
        'slippage-bps-input':
            'Execution slippage per traded notional, in basis points. This is separate from broker commission.',
        'regulatory-sell-bps-input':
            'Regulatory fees charged on sell notional, in basis points. Keep explicit instead of hiding them inside slippage.',
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
        '選股因子。勾選要使用的因子，右側下拉設定權重（— = 自動均權）。各因子的問號有詳細說明。\n' +
        'Stock-selection factors. Tick to enable, set weight on the right (— = auto/equal). Hover each factor\u2019s ? for details.');
}

function normalizeModels(raw) {
    let rows = [];
    if (Array.isArray(raw)) rows = raw;
    else if (raw && typeof raw === 'object') {
        rows = Object.entries(raw).map(([id, meta]) =>
            typeof meta === 'string' ? { id, label: meta } : Object.assign({ id }, meta || {}));
    }
    const normalized = rows.map((m, idx) => {
        if (typeof m === 'string') return { id: m, label: m, uses_factors: m === 'factor_composite' };
        const id = m.id || m.model_id || m.value || `model_${idx + 1}`;
        return {
            id,
            label: m.label || m.name || id,
            uses_factors: m.uses_factors !== false,
            description: m.description || '',
        };
    });
    if (!normalized.some(m => m.id === 'factor_composite')) {
        normalized.unshift({ id: 'factor_composite', label: 'Factor Composite', uses_factors: true });
    }
    return normalized;
}

function buildModelSelect(rawModels) {
    const sel = document.getElementById('model-select');
    if (!sel) return;
    const models = normalizeModels(rawModels);
    sel.innerHTML = '';
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.label;
        option.dataset.usesFactors = model.uses_factors ? '1' : '0';
        if (model.description) option.title = model.description;
        sel.appendChild(option);
    });
    sel.value = models.some(m => m.id === 'factor_composite') ? 'factor_composite' : models[0].id;
    onModelChange({ mark: false });
}

function getSelectedModelId() {
    return document.getElementById('model-select')?.value || 'factor_composite';
}

function selectedModelUsesFactors() {
    const sel = document.getElementById('model-select');
    const option = sel?.options[sel.selectedIndex];
    return !option || option.dataset.usesFactors !== '0';
}

function onModelChange(opts = {}) {
    const usesFactors = selectedModelUsesFactors();
    const wrap = document.getElementById('model-factors-wrap');
    if (wrap) wrap.style.display = usesFactors ? '' : 'none';
    const mwuButton = document.getElementById('mwu-toggle');
    const mwuHidden = document.getElementById('mwu-hidden');
    if (mwuButton) mwuButton.disabled = !usesFactors;
    if (!usesFactors && mwuHidden) {
        mwuHidden.value = '0';
        mwuButton.textContent = 'OFF';
        mwuButton.classList.remove('on');
    }
    document.querySelectorAll('#factor-checks-primary input, #factor-checks-primary select').forEach(el => {
        el.disabled = !usesFactors || (el.matches('select.factor-weight') &&
            (document.getElementById('mwu-hidden')?.value === '1' || !el.closest('.factor-chk')?.querySelector('input')?.checked));
    });
    if (opts.mark !== false) markCustomPreset({ preserveStrategy: false });
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
    sel.onchange = (e) => {
        e.preventDefault();
        markCustomPreset({ preserveStrategy: false });
    };
    // clicking the dropdown shouldn't toggle the parent <label>'s checkbox
    sel.onclick = (e) => e.stopPropagation();
    return sel;
}

function buildFactorChecks(factors, presets) {
    const primaryC = document.getElementById('factor-checks-primary');
    if (!primaryC) return;
    primaryC.innerHTML = '';
    const defaultOn = presets['Baseline 70/30'] || presets['Balanced'] || factors.slice(0, 2);
    factors.forEach(f => {
        const label = document.createElement('label');
        label.className = 'factor-chk apx-factor-switch apx-glass-switch';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = f;
        cb.checked = defaultOn.includes(f);
        cb.onchange = () => {
            markCustomPreset({ preserveStrategy: false });
            updateWeightDisabled();
            syncApxFactorSwitches(primaryC);
        };
        const name = document.createElement('span');
        name.className = 'factor-name';
        name.textContent = f;
        const wsel = _buildWeightSelect(f);
        label.appendChild(cb);
        label.appendChild(name);
        if (FACTOR_HELP[f]) addHelpDot(label, FACTOR_HELP[f]);
        label.appendChild(wsel);
        primaryC.appendChild(label);
    });
    updateWeightDisabled();
    initApxGlassStandard(primaryC);
}

// Any factor-checkbox or live-affecting param change drops the preset to custom
// so Apply Live picks up the exact on-screen configuration.
function markCustomPreset(options = {}) {
    if (_applyingPreset) return;
    const sel = document.getElementById('preset-select');
    if (sel) sel.value = '__modified__';
    if (options.preserveStrategy === false) _strategyPreset = null;
    _selectedPresetLabel = 'UNSAVED MODIFIED SET';
}

// Gray-out (disable) every weight dropdown while MWU is ON — MWU derives weights
// dynamically, so manual entry is meaningless. Also disable a factor's weight
// dropdown while that factor is unchecked.
function updateWeightDisabled() {
    const mwuOn = document.getElementById('mwu-hidden').value === '1';
    const usesFactors = selectedModelUsesFactors();
    document.querySelectorAll('.factor-chk').forEach(row => {
        const cb = row.querySelector('input[type=checkbox]');
        const sel = row.querySelector('select.factor-weight');
        if (cb) cb.disabled = !usesFactors;
        if (sel) sel.disabled = !usesFactors || mwuOn || !cb || !cb.checked;
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

function setHoldingPeriod(days) {
    const sel = document.getElementById('holding-period-select');
    const want = String(days || 5);
    if (!sel) return;
    if (![...sel.options].some(o => o.value === want)) {
        const option = document.createElement('option');
        option.value = want;
        option.textContent = `${want} trading days`;
        sel.appendChild(option);
    }
    sel.value = want;
}

function setTopN(value) {
    const sel = document.getElementById('top-n-select');
    const want = String(value || 30);
    if (!sel) return;
    if (![...sel.options].some(o => o.value === want)) {
        const option = document.createElement('option');
        option.value = want;
        option.textContent = want;
        sel.appendChild(option);
    }
    sel.value = want;
}

function setToggleState(buttonId, hiddenId, enabled) {
    const button = document.getElementById(buttonId);
    const hidden = document.getElementById(hiddenId);
    if (hidden) hidden.value = enabled ? '1' : '0';
    if (button) {
        button.textContent = enabled ? 'ON' : 'OFF';
        button.classList.toggle('on', !!enabled);
    }
}

function applyPresetRiskControls(risk = {}) {
    const regime = risk.regime_mode || 'off';
    const regimeSelect = document.getElementById('risk-regime-select');
    if (regimeSelect) regimeSelect.value = regime;
    const riskOff = document.getElementById('risk-off-leverage');
    if (riskOff && risk.risk_off_leverage != null) riskOff.value = String(risk.risk_off_leverage);
    setToggleState('risk-vol-toggle', 'risk-vol-hidden', !!risk.volatility_throttle);
    setToggleState('risk-liquidity-toggle', 'risk-liquidity-hidden', !!risk.liquidity_filter);
    setToggleState('risk-crowding-toggle', 'risk-crowding-hidden', !!risk.crowding_shock_guard);
    setToggleState('risk-sector-toggle', 'risk-sector-hidden', !!risk.sector_balance);
    onRiskChange();
}

function buildPresetSelect(presets) {
    const sel = document.getElementById('preset-select');
    if (!sel) return;
    sel.innerHTML = '<option value="__modified__">UNSAVED MODIFIED SET</option>';
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
    if (!_config || !name || name === '__modified__') {
        if (name === '__modified__') markCustomPreset();
        return;
    }
    _applyingPreset = true;

    // ── Full strategy preset (sleeves + leverage + winner-lock) ──────────────
    const sp = (_config.strategy_presets || {})[name];
    if (sp) {
        _strategyPreset = name;
        _selectedPresetLabel = name;
        const modelSel = document.getElementById('model-select');
        const modelId = sp.model_id || 'factor_composite';
        if (modelSel && [...modelSel.options].some(o => o.value === modelId)) {
            modelSel.value = modelId;
            onModelChange({ mark: false });
        }
        // check the union of all sleeve factors so the UI reflects what runs
        const union = [];
        const mergedW = {};
        (sp.sleeves || []).forEach(sl => (sl.factors || []).forEach(f => {
            if (!union.includes(f)) union.push(f);
            // surface each sleeve's within-sleeve weights on the dropdowns
            if (sl.weights && sl.weights[f] != null) mergedW[f] = sl.weights[f];
        }));
        document.querySelectorAll('#factor-checks-primary input[type=checkbox]').forEach(cb => {
            cb.checked = union.includes(cb.value);
        });
        setFactorWeights(mergedW);
        updateWeightDisabled();
        if (sp.leverage != null) {
            const lev = document.getElementById('leverage-slider');
            if (lev) {
                lev.value = sp.leverage;
                document.getElementById('leverage-val').textContent = parseFloat(sp.leverage).toFixed(1) + 'x';
                syncApxGlassRange(lev);
            }
        }
        if (sp.top_n != null) {
            setTopN(sp.top_n);
        }
        const presetHoldDays = sp.rebalance_days != null
            ? sp.rebalance_days
            : (sp.rebalance_frequency === 'daily' ? 1 : 5);
        setHoldingPeriod(presetHoldDays);
        setToggleState('mwu-toggle', 'mwu-hidden', !!sp.use_mwu);
        applyPresetRiskControls(sp.risk_management || {});
        syncApxFactorSwitches();
        if (sp.universe) {
            const uniSel = document.getElementById('universe-select');
            if (uniSel && [...uniSel.options].some(o => o.value === sp.universe)) { uniSel.value = sp.universe; onUniverseChange(); }
        }
        const sleeveStr = (sp.sleeves || []).map(s =>
            `${s.name} ${Math.round(s.alloc * 100)}%`).join(' / ');
        appendLog('info', `★ Strategy "${name}" — ${sp.label || ''}`);
        appendLog('info', `   leverage ${sp.leverage}x · top${sp.top_n} · sleeves: ${sleeveStr}`);
        _applyingPreset = false;
        return;
    }

    // ── Plain factor preset ──────────────────────────────────────────────────
    _strategyPreset = null;
    _selectedPresetLabel = name;
    const modelSel = document.getElementById('model-select');
    if (modelSel && [...modelSel.options].some(o => o.value === 'factor_composite')) {
        modelSel.value = 'factor_composite';
        onModelChange({ mark: false });
    }
    const factors = _config.factor_presets[name] || [];
    document.querySelectorAll('#factor-checks-primary input[type=checkbox]').forEach(cb => {
        cb.checked = factors.includes(cb.value);
    });

    // Static factor weights for this preset → populate the weight dropdowns.
    const wp = (_config.factor_weight_presets || {})[name] || null;
    setFactorWeights(wp);
    updateWeightDisabled();
    syncApxFactorSwitches();

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
        if (defaults.holding_period_days != null) setHoldingPeriod(defaults.holding_period_days);
    }
    appendLog('info', wp
        ? `Preset "${name}" — fixed weights ${Object.entries(wp).map(([k, v]) => `${k} ${Math.round(v * 100)}%`).join(' / ')}`
        : `Preset "${name}" applied.`);
    _applyingPreset = false;
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
    markCustomPreset();
    loadDataStatus();
}

function toggleField(btnId, hiddenId) {
    const btn = document.getElementById(btnId);
    const hidden = document.getElementById(hiddenId);
    const isOn = hidden.value === '1';
    hidden.value = isOn ? '0' : '1';
    btn.textContent = isOn ? 'OFF' : 'ON';
    btn.classList.toggle('on', !isOn);
    // MWU affects LIVE trading → drop to custom preset so Apply Live reflects it.
    markCustomPreset({ preserveStrategy: hiddenId !== 'mwu-hidden' });
    // MWU toggle grays/ungrays the per-factor weight dropdowns.
    if (hiddenId === 'mwu-hidden') updateWeightDisabled();
}

function initSettingsDirtyTracking() {
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;
    const handler = (event) => {
        const target = event.target;
        if (!target?.closest('[data-workspace-view="backtest"]')) return;
        if (target.id === 'preset-select' || target.id === 'result-history-select') return;
        if (target.type === 'hidden') return;
        markCustomPreset();
    };
    sidebar.addEventListener('input', handler);
    sidebar.addEventListener('change', handler);
}

function initWorkspaceTabs() {
    document.querySelectorAll('.header-tabs .tab[data-tab]').forEach(tab => {
        tab.addEventListener('click', () => switchWorkspace(tab.dataset.tab));
    });
    switchWorkspace('backtest', { load: false });
}

function switchWorkspace(view, opts = {}) {
    _activeView = view === 'live' ? 'live' : 'backtest';
    document.querySelectorAll('.header-tabs .tab[data-tab]').forEach(tab =>
        tab.classList.toggle('active', tab.dataset.tab === _activeView));
    document.querySelectorAll('[data-workspace-view]').forEach(panel => {
        const show = panel.dataset.workspaceView === _activeView;
        panel.classList.toggle('workspace-hidden', !show);
    });

    const holdingsTab = document.querySelector('.bottom-tab[data-btab="holdings"]');
    const weightsTab = document.querySelector('.bottom-tab[data-btab="weights"]');
    const logTab = document.querySelector('.bottom-tab[data-btab="log"]');
    if (holdingsTab) holdingsTab.textContent = _activeView === 'live' ? 'REAL POSITIONS' : 'HOLDINGS LOG';
    if (weightsTab) weightsTab.textContent = _activeView === 'live' ? 'CURRENT FACTOR WEIGHTS' : 'FACTOR WEIGHTS';
    if (logTab) logTab.textContent = _activeView === 'live' ? 'ORDERS / SYSTEM LOG' : 'SYSTEM LOG';

    if (opts.load === false) return;
    if (_activeView === 'live') {
        loadLiveDashboard({ persist: true });
    } else if (_lastBacktestResult) {
        renderBacktestResult(_lastBacktestResult);
    } else {
        renderHoldings([]);
        renderFactorWeights([], []);
        if (_chart) { _chart.remove(); _chart = null; _stratSeries = _liveSeries = _spySeries = null; }
        const placeholder = document.getElementById('chart-placeholder');
        if (placeholder) {
            placeholder.style.display = '';
            const sub = placeholder.querySelector('.ph-sub');
            if (sub) sub.textContent = 'Run or load a backtest to see the performance curve comparison';
        }
        const legend = document.getElementById('chart-legend');
        if (legend) legend.style.display = 'none';
    }
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
        const universe = document.getElementById('universe-select')?.value || 'spy_qqq';
        const params = new URLSearchParams({ universe });
        if (universe === 'custom') {
            const symbols = getCustomSymbols();
            if (symbols?.length) params.set('symbols', symbols.join(','));
        }
        const r = await fetch(`/data/status?${params.toString()}`);
        const d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
        const currentPct = d.fresh_pct ?? d.coverage_pct;
        setText('data-coverage-pct', currentPct != null ? Number(currentPct).toFixed(1) + '%' : '--');
        setText(
            'data-coverage-detail',
            `${d.covered || 0}/${d.total_symbols || 0} stored · ` +
            `${d.fresh_count ?? 0}/${d.total_symbols || 0} current · ${d.last_update || 'never'}`,
        );
    } catch (e) {
        setText('data-coverage-detail', 'unavailable');
    }
}

async function connectAPI() {
    const btn = document.getElementById('btn-connect');
    const account = getSelectedAccount();
    const mode = getSelectedAccountMode();
    const keyEnv = accountEnvKey(account);
    btn.disabled = true; btn.textContent = 'CONNECTING…';
    setApiStatus('connecting');
    appendLog('info', `Verifying ${account} (${mode}) via ${keyEnv}...`);
    try {
        const r = await fetch(`/live/status?account=${encodeURIComponent(account)}`);
        const d = await r.json();
        if (d.error) {
            appendLog('error', `Connection failed: ${d.error}`);
            setApiStatus('error');
        } else {
            const acct = d.account || {};
            const num = acct.account_number ? '••' + String(acct.account_number).slice(-4) : 'paper';
            setText('conn-account-info', `${account} ${mode.toUpperCase()} ${num}`);
            setText('account-env-info', keyEnv);
            const eq = parseFloat(acct.equity || 0);
            appendLog('success', `Connected ${account} (${mode}). Account ${num} - equity $${eq.toLocaleString()}`);
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
    const account = getSelectedAccount();
    if (btn) { btn.disabled = true; btn.textContent = 'FETCHING…'; }
    appendLog('info', `${mode === 'bulk' ? 'Full' : 'Incremental'} data fetch starting for ${account} (stores what is missing)...`);
    switchBtab('log');
    try {
        const universe = document.getElementById('universe-select')?.value || 'spy_qqq';
        const body = {
            mode,
            account,
            universe,
            symbols: universe === 'custom' ? getCustomSymbols() : null,
        };
        if (mode === 'bulk') body.start_date = '2015-01-01';
        const r = await fetch('/data/fetch', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const res = await r.json();
        if (!r.ok || res.error) throw new Error(res.error || `HTTP ${r.status}`);
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
    if (!selectedModelUsesFactors()) return [];
    const out = [];
    document.querySelectorAll('#factor-checks-primary input[type=checkbox]:checked').forEach(cb => out.push(cb.value));
    return out;
}

function getCustomSymbols() {
    const raw = document.getElementById('custom-symbols').value.trim();
    if (!raw) return null;
    return raw.split(/[\s,]+/).map(s => s.trim().toUpperCase()).filter(Boolean);
}

function onRiskChange() {
    const mode = document.getElementById('risk-regime-select')?.value || 'off';
    const hidden = document.getElementById('ema-kill-hidden');
    const wrap = document.getElementById('risk-off-lev-wrap');
    if (hidden) hidden.value = mode === 'cash' ? '1' : '0';
    if (wrap) wrap.style.display = mode === 'throttle' ? '' : 'none';
    markCustomPreset();
}

function getRiskManagement(overrides = null) {
    const base = {
        regime_mode: document.getElementById('risk-regime-select')?.value || 'off',
        risk_off_leverage: parseFloat(document.getElementById('risk-off-leverage')?.value || '0.75'),
        volatility_throttle: document.getElementById('risk-vol-hidden')?.value === '1',
        vol_target_pct: 0.25,
        vol_lookback: 20,
        liquidity_filter: document.getElementById('risk-liquidity-hidden')?.value === '1',
        min_price: 5.0,
        min_avg_dollar_vol: 20000000,
        crowding_shock_guard: document.getElementById('risk-crowding-hidden')?.value === '1',
        max_avg_range_pct: 0.12,
        sector_balance: document.getElementById('risk-sector-hidden')?.value === '1',
    };
    return Object.assign(base, overrides || {});
}

function buildBacktestBody(riskOverride = null) {
    const universe = document.getElementById('universe-select').value;
    const yr = getYearRange();
    const risk = getRiskManagement(riskOverride);
    return {
        model_id: getSelectedModelId(),
        ui_preset_label: _selectedPresetLabel,
        universe,
        symbols: universe === 'custom' ? getCustomSymbols() : null,
        start_date: yr.start_date,
        end_date: yr.end_date,
        active_factors: getActiveFactors(),
        capital: parseFloat(document.getElementById('capital-input').value) || 100000,
        leverage: parseFloat(document.getElementById('leverage-slider').value) || 1.0,
        use_mwu: document.getElementById('mwu-hidden').value === '1',
        top_n: parseInt(document.getElementById('top-n-select').value) || 30,
        holding_period_days: parseInt(document.getElementById('holding-period-select')?.value || '5') || 5,
        factor_weights: getFactorWeights(),
        strategy_preset: _strategyPreset || null,
        ema_kill_switch: risk.regime_mode === 'cash',
        risk_management: risk,
        commission_bps: Math.max(0, parseFloat(document.getElementById('commission-bps-input')?.value || '0')),
        slippage_bps: Math.max(0, parseFloat(document.getElementById('slippage-bps-input')?.value || '5')),
        regulatory_sell_bps: Math.max(0, parseFloat(document.getElementById('regulatory-sell-bps-input')?.value || '0')),
    };
}

async function runBacktest() {
    const factors = getActiveFactors();
    if (selectedModelUsesFactors() && factors.length === 0) {
        appendLog('warn', 'Select at least one factor before running backtest.');
        return;
    }
    const body = buildBacktestBody();

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
            renderBacktestResult(result);
            await saveBacktestHistory(body, result);
            appendLog('success', `Backtest complete — CAGR ${result.metrics?.cagr_pct?.toFixed(1)}% · Sharpe ${result.metrics?.sharpe?.toFixed(2)}`);
        }
    } catch (e) {
        stopProgressAnimation(false);
        appendLog('error', `Backtest request failed: ${e.message}`);
    } finally {
        btn.disabled = false; btn.textContent = 'EXECUTE BACKTEST';
    }
}

async function runSweepBacktest() {
    const factors = getActiveFactors();
    if (selectedModelUsesFactors() && factors.length === 0) {
        appendLog('warn', 'Select at least one factor before running sweep.');
        return;
    }
    const regimes = ['off', 'throttle', 'cash'];
    const bools = [false, true];
    const combos = [];
    const selectedHold = parseInt(document.getElementById('holding-period-select')?.value || '5') || 5;
    regimes.forEach(regime_mode => {
        bools.forEach(volatility_throttle => {
            bools.forEach(liquidity_filter => {
                bools.forEach(crowding_shock_guard => {
                    bools.forEach(sector_balance => {
                        combos.push({
                            regime_mode,
                            volatility_throttle,
                            liquidity_filter,
                            crowding_shock_guard,
                            sector_balance,
                            holding_period_days: selectedHold,
                        });
                    });
                });
            });
        });
    });

    const btn = document.getElementById('btn-sweep-backtest');
    const runBtn = document.getElementById('btn-run-backtest');
    if (btn) { btn.disabled = true; btn.textContent = 'SWEEPING...'; }
    if (runBtn) runBtn.disabled = true;
    switchBtab('log');
    appendLog('warn', `Sweep backtest starting: ${combos.length} risk combinations.`);

    const rows = [];
    try {
        for (let i = 0; i < combos.length; i++) {
            const combo = combos[i];
            const { holding_period_days, ...riskCombo } = combo;
            const body = buildBacktestBody(riskCombo);
            body.holding_period_days = holding_period_days;
            body.ema_kill_switch = riskCombo.regime_mode === 'cash';
            const label = `${combo.regime_mode}` +
                `${combo.volatility_throttle ? '+vol' : ''}` +
                `${combo.liquidity_filter ? '+liq' : ''}` +
                `${combo.crowding_shock_guard ? '+shock' : ''}` +
                `${combo.sector_balance ? '+sector' : ''}` +
                `+hold${holding_period_days}d`;
            appendLog('info', `Sweep ${i + 1}/${combos.length}: ${label}`);
            const r = await fetch('/backtest/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const result = await r.json();
            if (result.error) {
                appendLog('error', `Sweep ${label} failed: ${result.error}`);
                continue;
            }
            rows.push({ label, combo, result, metrics: result.metrics || {} });
        }
        rows.sort((a, b) => (b.metrics.calmar || -999) - (a.metrics.calmar || -999));
        appendLog('success', 'Sweep complete. Top combinations by Calmar:');
        rows.slice(0, 8).forEach((row, idx) => {
            const m = row.metrics;
            appendLog('info',
                `#${idx + 1} ${row.label} | Calmar ${fmtNum(m.calmar, 2)} | PF ${fmtNum(m.profit_factor, 2)} | CAGR ${fmtNum(m.cagr_pct, 1)}% | MaxDD ${fmtNum(m.max_dd_pct, 1)}%`);
        });
        if (rows[0]) {
            _lastBacktestResult = rows[0].result;
            renderBacktestResult(rows[0].result);
            await saveSweepHistory(rows);
        }
    } catch (e) {
        appendLog('error', `Sweep request failed: ${e.message}`);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'SWEEP BACKTEST'; }
        if (runBtn) runBtn.disabled = false;
    }
}

async function applyLive() {
    const factors = getActiveFactors();
    const account = getSelectedAccount();
    const mode = getSelectedAccountMode();
    const risk = getRiskManagement();
    if (selectedModelUsesFactors() && factors.length === 0) {
        appendLog('warn', 'Select at least one factor before applying this factor model live.');
        return;
    }
    const body = {
        account,
        model_id: getSelectedModelId(),
        strategy_preset: _strategyPreset || null,
        active_factors: factors,
        universe: document.getElementById('universe-select')?.value || 'spy_qqq',
        symbols: document.getElementById('universe-select')?.value === 'custom' ? getCustomSymbols() : null,
        leverage: parseFloat(document.getElementById('leverage-slider').value) || 1.0,
        top_n: parseInt(document.getElementById('top-n-select').value) || 20,
        rebalance_days: parseInt(document.getElementById('holding-period-select')?.value || '5') || 5,
        factor_weights: getFactorWeights(),
        use_mwu: document.getElementById('mwu-hidden').value === '1',
        ema_kill_switch: risk.regime_mode === 'cash',
        risk_management: risk,
    };

    const label = _selectedPresetLabel === 'UNSAVED MODIFIED SET'
        ? `UNSAVED MODIFIED SET${_strategyPreset ? ` based on "${_strategyPreset}" sleeves` : ''}`
        : _strategyPreset
        ? `strategy "${_strategyPreset}"`
        : (_selectedPresetLabel
            ? `factor preset "${_selectedPresetLabel}"`
            : `${getSelectedModelId()}${selectedModelUsesFactors() ? ` [${factors.join(', ')}]` : ''}`);
    const liveWarn = mode === 'live' ? '\n\nWARNING: this selected account is LIVE and can trade real money.' : '';
    if (!confirm(`Apply ${label} to account ${account} (${mode.toUpperCase()})?\n\nIt will be executed on the next eligible rebalance run.${liveWarn}`)) return;

    const btn = document.getElementById('btn-apply-live');
    if (btn) { btn.disabled = true; btn.textContent = 'APPLYING…'; }
    switchBtab('log');
    appendLog('warn', `Applying ${label} to ${account} (${mode}) live config...`);
    try {
        const r = await fetch('/live/apply', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const res = await r.json();
        if (res.error) {
            appendLog('error', `Apply live failed: ${res.error}`);
        } else {
            appendLog('success', `✓ Live strategy updated for ${account} (${mode}): ${res.description}`);
            appendLog('info', '→ Takes effect on the next eligible rebalance run.');
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
function renderBacktestResult(result) {
    if (!result) return;
    _lastBacktestResult = result;
    renderMetrics(result.metrics);
    renderBenchmarkMetrics(result.benchmark_metrics, result.benchmark);
    renderSectorExposure(result.sector_exposure);
    renderComparisonChart(result);
    renderHoldings(result.holdings || []);
    renderFactorWeights(result.factor_weights || [], result.params?.factors || result.params?.active_factors || []);
}

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
        { l: 'PF', v: fmtNum(m.profit_factor, 2), c: m.profit_factor >= 1.5 ? 'pos' : '' },
        { l: 'MAX DD', v: maxddStr, c: 'neg' },
        { l: 'DD FALL', v: m.max_dd_fall_days == null ? '--' : `${m.max_dd_fall_days}d`, c: 'neg' },
        { l: 'DD RECOVER', v: m.max_dd_recovery_days == null ? 'OPEN' : `${m.max_dd_recovery_days}d`, c: m.max_dd_recovery_days == null ? 'neg' : '' },
        { l: 'WIN RATE', v: fmtNum(m.win_rate_pct, 1) + '%', c: '' },
        { l: 'HOLD WIN', v: m.holding_win_rate_pct == null ? '--' : fmtNum(m.holding_win_rate_pct, 1) + '%', c: '' },
        { l: 'DAYS', v: (m.total_days || 0).toLocaleString(), c: '' },
    ];
    const totalCost = m.total_cost_dollars ?? m.total_cost ?? m.transaction_cost_dollars;
    const costPct = m.total_cost_pct_initial ?? m.total_cost_pct ?? m.transaction_cost_pct;
    const turnoverPct = m.annualized_one_way_turnover != null
        ? Number(m.annualized_one_way_turnover) * 100
        : (m.annual_turnover_pct ?? m.turnover_pct ?? m.average_turnover_pct);
    if (totalCost != null) rows.push({ l: 'TOTAL COST', v: '$' + fmtNum(totalCost, 2), c: 'neg' });
    if (costPct != null) rows.push({ l: 'COST / CAPITAL', v: fmtNum(costPct, 2) + '%', c: 'neg' });
    if (turnoverPct != null) rows.push({ l: 'ANNUAL TURNOVER · 1-WAY', v: fmtNum(turnoverPct, 1) + '%', c: '' });
    if (m.total_commission != null) rows.push({ l: 'COMMISSION', v: '$' + fmtNum(m.total_commission, 2), c: 'neg' });
    if (m.total_slippage != null) rows.push({ l: 'SLIPPAGE', v: '$' + fmtNum(m.total_slippage, 2), c: 'neg' });
    if (m.total_regulatory_fees != null) rows.push({ l: 'REGULATORY FEES', v: '$' + fmtNum(m.total_regulatory_fees, 2), c: 'neg' });
    if (breached) appendLog('warn', `MaxDD ${fmtNum(m.max_dd_pct,1)}% breaches the ${(liqDD*100).toFixed(1)}% liquidation drawdown at ${lev}x leverage (爆倉風險).`);
    grid.innerHTML = rows.map(r =>
        `<div class="metric-card"><div class="label">${r.l}</div><div class="value ${r.c}">${r.v}</div></div>`
    ).join('');
}

function renderBenchmarkMetrics(m, benchmark) {
    const panel = document.getElementById('benchmark-metrics-panel');
    const grid = document.getElementById('benchmark-metrics-grid');
    if (!panel || !grid) return;
    if (!m || !m.matched_days) {
        panel.style.display = 'none';
        grid.innerHTML = '';
        return;
    }

    panel.style.display = '';
    setText('benchmark-metrics-title', `BENCHMARK RELATIVE · ${benchmark || 'QQQ'}`);
    const signedPct = (v) => v == null ? '--' : `${v > 0 ? '+' : ''}${fmtNum(v, 1)}%`;
    const signedClass = (v) => v == null || v === 0 ? '' : (v > 0 ? 'pos' : 'neg');
    const captureClass = (v) => v != null && v < 100 ? 'pos' : '';
    const rows = [
        { l: 'QQQ CAGR', v: fmtNum(m.benchmark_cagr_pct, 1) + '%', c: '', t: 'Benchmark CAGR over matched strategy sessions.' },
        { l: 'EXCESS CAGR', v: signedPct(m.excess_cagr_pct), c: signedClass(m.excess_cagr_pct), t: 'Strategy CAGR minus benchmark CAGR.' },
        { l: 'ALPHA', v: signedPct(m.alpha_pct_annual), c: signedClass(m.alpha_pct_annual), t: 'Annualized Jensen alpha, assuming a zero risk-free rate.' },
        { l: 'BETA', v: fmtNum(m.beta, 2), c: '', t: 'Daily return sensitivity to the benchmark. 1.0 means benchmark-like market exposure.' },
        { l: 'TRACK ERR', v: fmtNum(m.tracking_error_pct, 1) + '%', c: '', t: 'Annualized volatility of strategy return minus benchmark return.' },
        { l: 'INFO RATIO', v: fmtNum(m.information_ratio, 2), c: signedClass(m.information_ratio), t: 'Annualized active return divided by tracking error.' },
        { l: 'CORRELATION', v: fmtNum(m.correlation, 2), c: '', t: 'Daily return correlation with the benchmark.' },
        { l: 'UP CAPTURE', v: fmtNum(m.upside_capture_pct, 0) + '%', c: m.upside_capture_pct > 100 ? 'pos' : '', t: 'Strategy participation on benchmark up days. Above 100% captures more upside.' },
        { l: 'DOWN CAPTURE', v: fmtNum(m.downside_capture_pct, 0) + '%', c: captureClass(m.downside_capture_pct), t: 'Strategy participation on benchmark down days. Lower is better; negative means it rose on average.' },
        { l: '1Y BEAT', v: m.rolling_1y_win_rate_pct == null ? '--' : fmtNum(m.rolling_1y_win_rate_pct, 0) + '%', c: m.rolling_1y_win_rate_pct > 50 ? 'pos' : '', t: `Share of ${m.rolling_1y_windows || 0} rolling 252-session windows that beat the benchmark.` },
        { l: 'LATEST 1Y', v: signedPct(m.latest_1y_excess_pct), c: signedClass(m.latest_1y_excess_pct), t: 'Latest 252-session strategy return minus benchmark return.' },
        { l: '3Y BEAT', v: m.rolling_3y_win_rate_pct == null ? '--' : fmtNum(m.rolling_3y_win_rate_pct, 0) + '%', c: m.rolling_3y_win_rate_pct > 50 ? 'pos' : '', t: `Share of ${m.rolling_3y_windows || 0} rolling 756-session windows that beat the benchmark.` },
        { l: 'LATEST 3Y', v: signedPct(m.latest_3y_excess_pct), c: signedClass(m.latest_3y_excess_pct), t: 'Latest 756-session strategy return minus benchmark return.' },
        { l: 'MATCHED DAYS', v: Number(m.matched_days).toLocaleString(), c: '', t: 'Trading sessions shared by strategy and benchmark.' },
    ];
    grid.innerHTML = rows.map(r =>
        `<div class="metric-card" title="${escHtml(r.t)}"><div class="label">${r.l}</div><div class="value ${r.c}">${r.v}</div></div>`
    ).join('');
}

function renderSectorExposure(exposure) {
    const panel = document.getElementById('sector-exposure-panel');
    const grid = document.getElementById('sector-exposure-grid');
    if (!panel || !grid) return;
    const rows = Object.entries(exposure || {});
    if (!rows.length) {
        panel.style.display = 'none';
        grid.innerHTML = '';
        return;
    }
    panel.style.display = '';
    grid.innerHTML = rows.map(([sector, value]) => {
        const pct = Number(value);
        const cls = pct > 25 ? 'neg' : (pct <= 15 ? 'pos' : '');
        return `<div class="metric-card"><div class="label">${escHtml(sector)}</div><div class="value ${cls}">${fmtNum(pct, 1)}%</div></div>`;
    }).join('');
}

function fmtNum(v, dec) {
    if (v == null || isNaN(v)) return '--';
    return Number(v).toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

function setChartView(mode) {
    _chartViewMode = mode || '1y';
    if (_activeView === 'live' && _currentLiveSnapshot?.equity_curve) {
        renderLiveChart(_currentLiveSnapshot.equity_curve, _currentLiveSnapshot.account_name || getSelectedAccount());
    } else if (_lastBacktestResult) {
        renderComparisonChart(_lastBacktestResult);
    }
}

function getChartViewMode() {
    const sel = document.getElementById('chart-view-select');
    return (sel && sel.value) || _chartViewMode || '1y';
}

function shiftDate(dateStr, years) {
    const d = new Date(`${dateStr}T00:00:00Z`);
    d.setUTCFullYear(d.getUTCFullYear() - years);
    return d.toISOString().slice(0, 10);
}

function chartStartDate(curve, mode) {
    if (!curve || !curve.length || mode === 'full') return null;
    const last = curve[curve.length - 1].date;
    if (mode === 'ytd') return `${String(last).slice(0, 4)}-01-01`;
    if (mode === '2y') return shiftDate(last, 2);
    if (mode === '3y') return shiftDate(last, 3);
    return shiftDate(last, 1);
}

function curveToChartData(curve, mode) {
    if (!curve || !curve.length) return [];
    const start = chartStartDate(curve, mode);
    let sliced = start ? curve.filter(p => p.date >= start) : curve.slice();
    if (sliced.length < 2) sliced = curve.slice(Math.max(0, curve.length - 252));
    if (mode === 'full') {
        return sliced.map(p => ({ time: p.date, value: p.value }));
    }
    const base = parseFloat(sliced[0]?.value || 0);
    if (!(base > 0)) return [];
    return sliced.map(p => ({ time: p.date, value: Math.round((p.value / base) * 10000) / 100 }));
}

function updateChartLegend(mode, benchmark) {
    const indexed = mode !== 'full';
    const setLegend = (id, swatchStyle, label) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = `<span class="swatch" style="${swatchStyle}"></span>${escHtml(label)}`;
    };
    setLegend('legend-strategy', 'background:#00e5a0;', indexed ? 'STRATEGY (INDEXED)' : 'STRATEGY (BACKTEST)');
    setLegend('legend-benchmark', 'background:#ffa726;border-top:1px dashed #ffa726;', indexed ? `${benchmark || 'QQQ'} (INDEXED)` : `${benchmark || 'QQQ'} BENCHMARK`);
    setLegend('legend-live', 'background:#64dcff;', indexed ? 'LIVE (INDEXED)' : 'LIVE ACCOUNT');
}

// ── Comparison chart: backtest + live + SPY ───────────────────────────────────────
function renderComparisonChart(result) {
    const container = document.getElementById('chart-container');
    document.getElementById('chart-placeholder').style.display = 'none';
    document.getElementById('chart-legend').style.display = 'flex';
    const mode = getChartViewMode();
    const benchLabel = result.benchmark || 'QQQ';
    updateChartLegend(mode, benchLabel);
    const ls = document.getElementById('legend-strategy');
    const ll = document.getElementById('legend-live');
    const lb = document.getElementById('legend-benchmark');
    if (ls) ls.style.display = '';
    if (ll) ll.style.display = '';
    if (lb) lb.style.display = '';

    if (_chart) { _chart.remove(); _chart = null; _stratSeries = _liveSeries = _spySeries = null; }

    _chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        ...chartThemeOptions(),
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        timeScale: { ...chartThemeOptions().timeScale, timeVisible: false, secondsVisible: false },
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

    _stratSeries = _chart.addLineSeries({ color: '#00e5a0', lineWidth: 2, title: 'Strategy' });
    _spySeries = _chart.addLineSeries({ color: '#ffa726', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, title: benchLabel });
    _liveSeries = _chart.addLineSeries({ color: '#64dcff', lineWidth: 2, title: 'Live' });

    if (result.equity_curve?.length)
        _stratSeries.setData(curveToChartData(result.equity_curve, mode));
    const benchCurve = result.benchmark_curve || result.spy_curve;
    if (benchCurve?.length)
        _spySeries.setData(curveToChartData(benchCurve, mode));

    _chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
        if (_chart) _chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    });
    ro.observe(container);

    // Best-effort live account overlay (read-only portfolio history)
    overlayLiveCurve();
}

async function overlayLiveCurve() {
    const account = getSelectedAccount();
    try {
        const r = await fetch(`/live/equity?account=${encodeURIComponent(account)}&period=all`);
        const d = await r.json();
        const curve = d.equity_curve || [];
        if (!curve.length || !_liveSeries) {
            if (_liveSeries) appendLog('info', `No live account history to overlay for ${account}.`);
            return;
        }

        // ── Time + money alignment ───────────────────────────────────────────
        // The live account starts whenever it was funded (e.g. 2026) with its
        // own real dollar balance. To compare it against a backtest that began
        // earlier (e.g. 2022) we re-base the live curve: it starts at the value
        // the *strategy* (backtest) curve had on the live start date, then
        // tracks the live account's actual % returns from there. This way both
        // lines share the same money scale and connect in time.
        const mode = getChartViewMode();
        if (mode !== 'full') {
            const liveData = curveToChartData(curve, mode);
            if (liveData.length) _liveSeries.setData(liveData);
            appendLog('info', `Live overlay for ${account}: ${curve.length} days, ${mode.toUpperCase()} indexed view.`);
            return;
        }

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
            appendLog('info', `Live overlay for ${account}: ${curve.length} days, re-based to strategy equity on ${liveStart} (x${scale.toFixed(3)}).`);
        } else {
            appendLog('info', `Live account overlay for ${account}: ${curve.length} days.`);
        }
    } catch (e) {
        // live overlay is optional; ignore failures
    }
}

async function fetchJsonOptional(url, fallback) {
    try {
        const response = await fetch(url);
        if (!response.ok) return fallback;
        const value = await response.json();
        return value == null ? fallback : value;
    } catch (e) {
        return fallback;
    }
}

async function loadLiveDashboard(opts = {}) {
    const account = getSelectedAccount();
    const button = document.getElementById('btn-refresh-live');
    if (button) { button.disabled = true; button.textContent = 'REFRESHING…'; }
    try {
        const q = encodeURIComponent(account);
        const [status, equity, activitiesRaw, ordersRaw, performance] = await Promise.all([
            fetchJsonOptional(`/live/status?account=${q}`, {}),
            fetchJsonOptional(`/live/equity?account=${q}&period=all`, { equity_curve: [] }),
            fetchJsonOptional(`/live/activities?account=${q}&limit=1000`, []),
            fetchJsonOptional(`/live/orders?account=${q}&limit=500`, []),
            fetchJsonOptional(`/live/performance?account=${q}`, {}),
        ]);
        if (account !== getSelectedAccount()) return;
        if (status.error) throw new Error(status.error);
        const activities = Array.isArray(activitiesRaw) ? activitiesRaw : (activitiesRaw.activities || []);
        const orders = Array.isArray(ordersRaw) ? ordersRaw : (ordersRaw.orders || []);
        const localAudit = Array.isArray(ordersRaw?.local_audit) ? ordersRaw.local_audit : [];
        const snapshot = {
            account_name: account,
            captured_at: new Date().toISOString(),
            status,
            equity_curve: equity.equity_curve || [],
            activities,
            orders,
            local_audit: localAudit,
            performance: performance || {},
        };
        _currentLiveSnapshot = snapshot;
        renderLiveSnapshot(snapshot);
        if (opts.persist !== false) await saveLiveHistory(snapshot);
        appendLog('success', `Live dashboard refreshed for ${account}: ${(status.positions || []).length} positions, ${orders.length + localAudit.length || activities.length} execution records.`);
    } catch (e) {
        renderLiveUnavailable(e.message);
        appendLog('error', `Live dashboard unavailable: ${e.message}`);
    } finally {
        if (button) { button.disabled = false; button.textContent = 'REFRESH LIVE'; }
    }
}

function renderLiveUnavailable(message) {
    const grid = document.getElementById('live-metrics-grid');
    if (grid) grid.innerHTML = `<div class="empty-state">${escHtml(message || 'Live account data is unavailable.')}</div>`;
    renderLivePositions([]);
    renderLiveFactorWeights({});
}

function renderLiveSnapshot(snapshot) {
    if (!snapshot) return;
    _currentLiveSnapshot = snapshot;
    const status = snapshot.status || {};
    const account = status.account || {};
    const config = status.live_config || {};
    const performance = snapshot.performance || {};
    const positions = status.positions || [];
    const trackerHistory = Array.isArray(status.tracker_history) ? status.tracker_history : [];
    const latestAdjustedTracker = trackerHistory.slice().reverse().find(row =>
        row?.total_pnl_basis === 'cash_flow_adjusted_linked_return'
    );
    const unrealizedFallback = positions.reduce((sum, p) => sum + (Number(p.unrealized_pl) || 0), 0);
    const realized = performance.final_pnl ?? performance.realized_pnl;
    const unrealized = performance.unrealized_pnl ?? unrealizedFallback;
    const metricRows = [
        { l: 'EQUITY', v: '$' + fmtNum(account.equity, 2), c: '' },
        { l: 'FINAL PNL · REALIZED', v: realized == null ? '--' : '$' + fmtNum(realized, 2), c: realized == null ? '' : (realized >= 0 ? 'pos' : 'neg') },
        { l: 'UNREALIZED', v: '$' + fmtNum(unrealized, 2), c: unrealized >= 0 ? 'pos' : 'neg' },
        { l: 'CASH', v: '$' + fmtNum(account.cash, 2), c: '' },
        { l: 'BUYING POWER', v: '$' + fmtNum(account.buying_power, 2), c: '' },
        { l: 'POSITIONS', v: String(positions.length), c: '' },
    ];
    if (latestAdjustedTracker) {
        const linkedReturn = Number(latestAdjustedTracker.total_pnl_pct) || 0;
        const baselineEquity = Number(latestAdjustedTracker.performance_baseline?.equity) || 0;
        const netCashFlow = Number(latestAdjustedTracker.net_cash_flow_since_baseline) || 0;
        const trackedEquity = Number(latestAdjustedTracker.equity) || 0;
        const strategyNetPnl = trackedEquity - baselineEquity - netCashFlow;
        metricRows.push({
            l: 'STRATEGY TWR · CASH-FLOW ADJ',
            v: fmtNum(linkedReturn * 100, 2) + '%',
            c: linkedReturn >= 0 ? 'pos' : 'neg',
        });
        metricRows.push({
            l: 'STRATEGY NET PNL · CASH-FLOW ADJ',
            v: '$' + fmtNum(strategyNetPnl, 2),
            c: strategyNetPnl >= 0 ? 'pos' : 'neg',
        });
    }
    if (performance.fill_count != null) metricRows.push({ l: 'FILLS IN SCOPE', v: String(performance.fill_count), c: '' });
    const grid = document.getElementById('live-metrics-grid');
    if (grid) {
        grid.innerHTML = metricRows.map(row =>
            `<div class="metric-card"><div class="label">${row.l}</div><div class="value ${row.c}">${row.v}</div></div>`).join('');
        if (performance.note) grid.innerHTML += `<div class="empty-state" style="grid-column:1/-1;padding:8px;">${escHtml(performance.note)}</div>`;
    }

    setText('live-preset-value', config.strategy_preset || 'Custom');
    setText('live-model-value', config.model_id || 'factor_composite');
    setText('live-factors-value', (config.active_factors || []).join(', ') || '--');
    setText('live-strategy-value', `Top ${config.top_n ?? '--'} · ${config.rebalance_days ?? '--'}d · ${config.leverage ?? '--'}x`);
    setText('live-rebalance-value', status.last_rebalance || '--');
    const market = status.market || {};
    setText('live-market-value', market.is_open ? 'OPEN' : `CLOSED${market.next_open ? ` · ${String(market.next_open).slice(0, 16)}` : ''}`);

    renderLivePositions(positions);
    renderLiveFactorWeights(config, status.effective_factor_weights || null);
    renderLiveOrders(snapshot.orders || [], snapshot.activities || [], snapshot.local_audit || [], status.tracker_history || []);
    renderLiveChart(snapshot.equity_curve || [], snapshot.account_name || getSelectedAccount());
}

function renderLivePositions(positions) {
    const head = document.getElementById('holdings-head-row');
    const body = document.getElementById('holdings-tbody');
    if (!head || !body) return;
    head.innerHTML = '<th>SYMBOL</th><th>QTY</th><th>MARKET VALUE</th><th>AVG ENTRY</th><th>LAST</th><th>UNREALIZED</th><th>P/L</th>';
    body.innerHTML = '';
    if (!positions.length) {
        body.innerHTML = '<tr><td colspan="7" class="empty-state">No open positions returned by the broker.</td></tr>';
        return;
    }
    positions.slice().sort((a, b) => Math.abs(Number(b.market_value) || 0) - Math.abs(Number(a.market_value) || 0)).forEach(position => {
        const pnl = Number(position.unrealized_pl) || 0;
        const pct = Number(position.pl_pct) || 0;
        const row = document.createElement('tr');
        row.innerHTML = `<td style="color:var(--green);">${escHtml(position.symbol || '')}</td>` +
            `<td>${fmtNum(position.qty, 4)}</td><td>$${fmtNum(position.market_value, 2)}</td>` +
            `<td>$${fmtNum(position.avg_entry, 2)}</td><td>$${fmtNum(position.current_price, 2)}</td>` +
            `<td class="${pnl >= 0 ? 'pos' : 'neg'}">$${fmtNum(pnl, 2)}</td>` +
            `<td class="${pct >= 0 ? 'pos' : 'neg'}">${fmtNum(pct, 2)}%</td>`;
        body.appendChild(row);
    });
}

function renderLiveFactorWeights(config, effectiveWeights = null) {
    const configuredFactors = config.active_factors || [];
    const factors = configuredFactors.length
        ? configuredFactors
        : Object.keys(effectiveWeights || config.factor_weights || {});
    let weights = effectiveWeights || config.factor_weights || {};
    if (!Object.keys(weights).length && factors.length) {
        weights = Object.fromEntries(factors.map(f => [f, 1 / factors.length]));
    }
    renderFactorWeights(factors.length ? [Object.assign({ date: String(config.saved_at || '').slice(0, 10) || 'CURRENT' }, weights)] : [], factors);
    if (!factors.length) {
        const body = document.getElementById('weights-tbody');
        if (body) body.innerHTML = '<tr><td class="empty-state">No factor weights: the current model may not use factors.</td></tr>';
    }
}

function renderLiveOrders(orders, activities, localAudit = [], trackerHistory = []) {
    const container = document.getElementById('log-container');
    if (!container) return;
    container.querySelectorAll('[data-live-order]').forEach(el => el.remove());
    const add = (html, cls = '') => {
        const line = document.createElement('div');
        line.dataset.liveOrder = '1';
        line.className = `log-line ${cls}`;
        line.innerHTML = html;
        container.prepend(line);
    };
    const rows = [...orders, ...localAudit];
    if (!rows.length && activities.length) rows.push(...activities);
    if (!rows.length) {
        add('<span class="log-msg">No broker order/fill history returned. Pending and processed orders appear here when /live/orders is available.</span>');
    } else {
        rows.slice().reverse().forEach(order => {
            const detail = order.details && typeof order.details === 'object' ? order.details : {};
            const status = String(order.status || detail.status || ((order.price ?? detail.price) != null ? 'filled' : 'unknown')).toLowerCase();
            const safeStatus = status.replace(/[^a-z_-]/g, '');
            const when = order.recorded_at || order.created_at || order.submitted_at || order.filled_at || order.time || order.date || '--';
            const price = order.filled_avg_price ?? order.price ?? detail.filled_avg_price ?? detail.price;
            const symbol = order.symbol || detail.symbol || order.event_type || '--';
            const side = order.side || detail.side || (order.event_type ? 'event' : '--');
            const qty = order.qty ?? detail.qty;
            add(`<span class="order-line"><span>${escHtml(String(when).replace('T', ' ').slice(0, 19))}</span>` +
                `<span>${escHtml(symbol)}</span><span>${escHtml(side)}</span>` +
                `<span>${qty == null ? '--' : fmtNum(qty, 4)}${price != null ? ` @ $${fmtNum(price, 2)}` : ''}</span>` +
                `<span class="order-status ${safeStatus}">${escHtml(status.toUpperCase())}</span></span>`);
        });
        add(`<span class="log-msg">BROKER EXECUTION HISTORY · ${rows.length} records · pending and processed</span>`, 'log-info');
    }
    trackerHistory.slice().reverse().forEach(event => {
        const when = event.timestamp || event.recorded_at || event.date || '--';
        const basis = event.total_pnl_basis === 'cash_flow_adjusted_linked_return'
            ? 'cash-flow adjusted'
            : 'legacy unadjusted';
        const hasBrokerDayPnl = Number.isFinite(Number(event.broker_calendar_day_cash_adjusted_pnl));
        const pnlValue = hasBrokerDayPnl
            ? event.broker_calendar_day_cash_adjusted_pnl
            : event.observation_pnl ?? event.day_pnl;
        const pnlLabel = hasBrokerDayPnl ? 'calendar-day P/L' : 'observation P/L';
        const estimate = event.return_calculation?.is_estimate ? ' estimate' : '';
        add(`<span class="log-msg">TRACKER ${escHtml(String(when).replace('T', ' ').slice(0, 19))}` +
            ` · equity $${fmtNum(event.equity, 2)} · ${pnlLabel} $${fmtNum(pnlValue, 2)}` +
            ` · total ${fmtNum((Number(event.total_pnl_pct) || 0) * 100, 2)}%` +
            ` · ${escHtml(basis + estimate)}</span>`, 'log-info');
    });
}

function renderLiveChart(curve, account) {
    const container = document.getElementById('chart-container');
    const placeholder = document.getElementById('chart-placeholder');
    const legend = document.getElementById('chart-legend');
    if (!curve?.length) {
        if (_chart) { _chart.remove(); _chart = null; _stratSeries = _liveSeries = _spySeries = null; }
        if (placeholder) {
            placeholder.style.display = '';
            const sub = placeholder.querySelector('.ph-sub');
            if (sub) sub.textContent = `No live equity history returned for ${account}.`;
        }
        if (legend) legend.style.display = 'none';
        return;
    }
    if (placeholder) placeholder.style.display = 'none';
    if (legend) legend.style.display = 'flex';
    const ls = document.getElementById('legend-strategy');
    const lb = document.getElementById('legend-benchmark');
    const ll = document.getElementById('legend-live');
    if (ls) ls.style.display = 'none';
    if (lb) lb.style.display = 'none';
    if (ll) {
        ll.style.display = '';
        ll.innerHTML = `<span class="swatch" style="background:#64dcff;"></span>${escHtml(account)} LIVE${getChartViewMode() === 'full' ? '' : ' (INDEXED)'}`;
    }
    if (_chart) { _chart.remove(); _chart = null; _stratSeries = _liveSeries = _spySeries = null; }
    _chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        ...chartThemeOptions(),
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        timeScale: { ...chartThemeOptions().timeScale, timeVisible: false, secondsVisible: false },
    });
    _liveSeries = _chart.addLineSeries({ color: '#64dcff', lineWidth: 2, title: `${account} Live` });
    _liveSeries.setData(curveToChartData(curve, getChartViewMode()));
    _chart.timeScale().fitContent();
    new ResizeObserver(() => {
        if (_chart) _chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    }).observe(container);
}

// ── Tables ──────────────────────────────────────────────────────────────────────
function renderHoldings(rows) {
    const tbody = document.getElementById('holdings-tbody');
    const thead = document.getElementById('holdings-head-row');
    if (!tbody) return;
    if (thead) thead.innerHTML = '<th style="width:90px;">DATE</th><th>PORTFOLIO <span class="lbl-hint">(longest-held first)</span></th>';
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

// ── Persistent unified result history ───────────────────────────────────────────
// IndexedDB keeps full curves/holdings without localStorage's small quota. A small
// localStorage fallback is retained for browsers where IndexedDB is unavailable.
function initHistoryStore() {
    return new Promise(resolve => {
        if (!window.indexedDB) { resolve(null); return; }
        const request = indexedDB.open(HISTORY_DB_NAME, HISTORY_DB_VERSION);
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(HISTORY_STORE)) {
                const store = db.createObjectStore(HISTORY_STORE, { keyPath: 'id' });
                store.createIndex('created_at', 'created_at');
            }
        };
        request.onsuccess = () => { _historyDb = request.result; resolve(_historyDb); };
        request.onerror = () => { _historyDb = null; resolve(null); };
    });
}

function localHistoryFallback() {
    try { return JSON.parse(localStorage.getItem('ancserAPXResultHistory') || '[]'); }
    catch (e) { return []; }
}

function readLocalHistory() {
    if (!_historyDb) return Promise.resolve(localHistoryFallback());
    return new Promise(resolve => {
        const request = _historyDb.transaction(HISTORY_STORE, 'readonly').objectStore(HISTORY_STORE).getAll();
        request.onsuccess = () => resolve(request.result || []);
        request.onerror = () => resolve([]);
    });
}

function putLocalHistory(record) {
    if (!_historyDb) {
        const rows = localHistoryFallback().filter(row => row.id !== record.id);
        rows.push(record);
        try {
            // Quota-safe fallback: preserve the newest complete records.
            localStorage.setItem('ancserAPXResultHistory', JSON.stringify(rows.slice(-20)));
        } catch (e) {
            try { localStorage.setItem('ancserAPXResultHistory', JSON.stringify(rows.slice(-5))); } catch (_) {}
        }
        return Promise.resolve();
    }
    return new Promise(resolve => {
        const tx = _historyDb.transaction(HISTORY_STORE, 'readwrite');
        tx.objectStore(HISTORY_STORE).put(record);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
    });
}

function deleteLocalHistory(id) {
    if (!_historyDb) {
        const rows = localHistoryFallback().filter(row => row.id !== id);
        try { localStorage.setItem('ancserAPXResultHistory', JSON.stringify(rows)); } catch (e) {}
        return Promise.resolve();
    }
    return new Promise(resolve => {
        const tx = _historyDb.transaction(HISTORY_STORE, 'readwrite');
        tx.objectStore(HISTORY_STORE).delete(id);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
    });
}

async function fetchServerHistory() {
    const payload = await fetchJsonOptional('/backtest/history?limit=500', []);
    const rows = Array.isArray(payload) ? payload : (payload.runs || payload.history || []);
    return rows.filter(Boolean).map((run, idx) => ({
        id: `server:${run.run_id || run.id || `${run.created_at || 'unknown'}:${idx}`}`,
        kind: run.kind || 'backtest',
        created_at: run.created_at || run.timestamp || new Date(0).toISOString(),
        label: run.label || run.name || `Backtest ${run.run_id || idx + 1}`,
        request: run.request || run.params || null,
        result: run.result || run.output || run,
        source: 'server',
    }));
}

function hiddenServerHistoryIds() {
    try { return new Set(JSON.parse(localStorage.getItem('ancserAPXHiddenServerHistory') || '[]')); }
    catch (e) { return new Set(); }
}

async function refreshHistoryList() {
    const [localRows, serverRows] = await Promise.all([readLocalHistory(), fetchServerHistory()]);
    const hidden = hiddenServerHistoryIds();
    const byId = new Map();
    [...serverRows, ...localRows].forEach(row => {
        if (!row?.id || hidden.has(row.id)) return;
        byId.set(row.id, row);
    });
    _historyRecords = [...byId.values()].sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
    const select = document.getElementById('result-history-select');
    if (!select) return;
    const selected = select.value;
    select.innerHTML = '';
    if (!_historyRecords.length) {
        select.innerHTML = '<option value="">No saved results yet</option>';
        return;
    }
    _historyRecords.forEach(record => {
        const option = document.createElement('option');
        option.value = record.id;
        const stamp = String(record.created_at || '').replace('T', ' ').slice(0, 16);
        const tag = record.kind === 'live' ? 'LIVE' : (record.kind === 'sweep' ? 'SWEEP' : 'BT');
        option.textContent = `[${tag}] ${stamp} · ${record.label || record.id}`;
        select.appendChild(option);
    });
    if ([...select.options].some(option => option.value === selected)) select.value = selected;
}

async function saveHistoryRecord(record) {
    await putLocalHistory(record);
    await refreshHistoryList();
    const select = document.getElementById('result-history-select');
    if (select && [...select.options].some(option => option.value === record.id)) select.value = record.id;
}

function historyRunLabel(body, result) {
    const model = body.ui_preset_label || body.strategy_preset || body.model_id || 'factor_composite';
    const factors = body.active_factors?.length ? ` · ${body.active_factors.join('+')}` : '';
    const cagr = result.metrics?.cagr_pct;
    return `${model}${factors} · Top${body.top_n} · ${body.holding_period_days}d${cagr == null ? '' : ` · CAGR ${fmtNum(cagr, 1)}%`}`;
}

async function saveBacktestHistory(body, result) {
    const created = new Date().toISOString();
    const id = result.run_id ? `server:${result.run_id}` : `local:bt:${created}:${Math.random().toString(36).slice(2, 8)}`;
    await saveHistoryRecord({
        id, kind: 'backtest', created_at: created,
        label: historyRunLabel(body, result), request: body, result,
        source: result.run_id ? 'server' : 'browser',
    });
}

async function saveSweepHistory(rows) {
    if (!rows.length) return;
    const created = new Date().toISOString();
    const champion = rows[0];
    await saveHistoryRecord({
        id: `local:sweep:${created}:${Math.random().toString(36).slice(2, 8)}`,
        kind: 'sweep', created_at: created,
        label: `${champion.label} · ${rows.length} variants · best Calmar ${fmtNum(champion.metrics?.calmar, 2)}`,
        result: champion.result,
        sweep_summary: rows.map(row => ({ label: row.label, combo: row.combo, metrics: row.metrics })),
        source: 'browser',
    });
}

async function saveLiveHistory(snapshot) {
    const lastDate = snapshot.equity_curve?.at(-1)?.date || snapshot.captured_at.slice(0, 10);
    const realized = snapshot.performance?.final_pnl ?? snapshot.performance?.realized_pnl;
    await saveHistoryRecord({
        id: `local:live:${snapshot.account_name}:${lastDate}`,
        kind: 'live', created_at: snapshot.captured_at,
        label: `${snapshot.account_name} · ${lastDate}${realized == null ? '' : ` · realized $${fmtNum(realized, 2)}`}`,
        snapshot,
        source: 'browser',
    });
}

function loadSelectedHistory() {
    const id = document.getElementById('result-history-select')?.value;
    const record = _historyRecords.find(row => row.id === id);
    if (!record) return;
    if (record.kind === 'live') {
        const snapshot = record.snapshot || record.result;
        if (!snapshot) return;
        setSelectedAccount(snapshot.account_name || getSelectedAccount(), { silent: true });
        switchWorkspace('live', { load: false });
        renderLiveSnapshot(snapshot);
        appendLog('info', `Loaded saved live snapshot: ${record.label}`);
        return;
    }
    const result = record.result;
    if (!result?.equity_curve) {
        appendLog('warn', `Saved record has no renderable equity curve: ${record.label}`);
        return;
    }
    switchWorkspace('backtest', { load: false });
    renderBacktestResult(result);
    appendLog('info', `Loaded saved ${record.kind} result: ${record.label}`);
}

async function deleteSelectedHistory() {
    const select = document.getElementById('result-history-select');
    const id = select?.value;
    if (!id) return;
    if (id.startsWith('server:')) {
        const hidden = hiddenServerHistoryIds();
        hidden.add(id);
        try { localStorage.setItem('ancserAPXHiddenServerHistory', JSON.stringify([...hidden])); } catch (e) {}
    }
    await deleteLocalHistory(id);
    await refreshHistoryList();
}

// ── Helpers ─────────────────────────────────────────────────────────────────────
function setText(id, t) { const el = document.getElementById(id); if (el) el.textContent = t; }

function switchBtab(tab) {
    const t = document.querySelector(`.bottom-tab[data-btab="${tab}"]`);
    if (t) t.click();
}
