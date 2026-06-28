/* ancserFX — ancserTPX skin over the ancserFX factor backend.
   Backtest-only. Central chart overlays backtest + live + SPY curves. */
'use strict';

// ── State ───────────────────────────────────────────────────────────────────
let _ws = null;
let _config = null;
let _presetWeights = null;   // static factor weights for current preset (e.g. v1.5S 70/30)
let _strategyPreset = null;  // active full-strategy preset name (e.g. "Claude #1")
let _activeBtab = 'holdings';
let _chart = null;
let _stratSeries = null;
let _liveSeries = null;
let _spySeries = null;
let _lastBacktestResult = null;
let _progressInterval = null;
let _connected = false;

// ── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    startClock();
    initWebSocket();
    loadConfig();
    populateYearSelects();
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
    }
}

function buildFactorChecks(factors, presets) {
    const c = document.getElementById('factor-checks');
    if (!c) return;
    c.innerHTML = '';
    const defaultOn = presets['Balanced'] || factors.slice(0, 3);
    factors.forEach(f => {
        const label = document.createElement('label');
        label.className = 'factor-chk';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = f;
        cb.checked = defaultOn.includes(f);
        cb.onchange = () => { document.getElementById('preset-select').value = ''; _presetWeights = null; _strategyPreset = null; };
        label.appendChild(cb);
        label.appendChild(document.createTextNode(' ' + f));
        c.appendChild(label);
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
    const defaultPreset = presets['v1.5S 70/30 Top20'] ? 'v1.5S 70/30 Top20' : 'Balanced';
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
        _presetWeights = null;
        // check the union of all sleeve factors so the UI reflects what runs
        const union = [];
        (sp.sleeves || []).forEach(sl => (sl.factors || []).forEach(f => {
            if (!union.includes(f)) union.push(f);
        }));
        document.querySelectorAll('#factor-checks input[type=checkbox]').forEach(cb => {
            cb.checked = union.includes(cb.value);
        });
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
            `${s.name} ${Math.round(s.alloc * 100)}%${s.winner_lock ? '+lock' : ''}`).join(' / ');
        const wl = sp.winner_lock || {};
        appendLog('info', `★ Strategy "${name}" — ${sp.label || ''}`);
        appendLog('info', `   leverage ${sp.leverage}x · top${sp.top_n} · sleeves: ${sleeveStr}`);
        if (Object.keys(wl).length) appendLog('info',
            `   winner-lock: profit≥${Math.round((wl.profit_lock||0)*100)}% · cap ${Math.round((wl.max_weight||0)*100)}% · rank≤${wl.lock_rank}`);
        return;
    }

    // ── Plain factor preset ──────────────────────────────────────────────────
    _strategyPreset = null;
    const factors = _config.factor_presets[name] || [];
    document.querySelectorAll('#factor-checks input[type=checkbox]').forEach(cb => {
        cb.checked = factors.includes(cb.value);
    });

    // Static factor weights for this preset (e.g. v1.5S 70/30). null → equal/MWU.
    const wp = (_config.factor_weight_presets || {})[name] || null;
    _presetWeights = wp;

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

// ── Year selects ──────────────────────────────────────────────────────────────────
function populateYearSelects() {
    const startSel = document.getElementById('start-year-select');
    const endSel = document.getElementById('end-year-select');
    const cur = new Date().getFullYear();
    for (let y = 2015; y <= cur; y++) {
        const o1 = document.createElement('option');
        o1.value = `${y}-01-01`; o1.textContent = y;
        if (y === 2020) o1.selected = true;
        startSel.appendChild(o1);
        const o2 = document.createElement('option');
        o2.value = `${y}-12-31`; o2.textContent = y;
        if (y === cur) o2.selected = true;
        endSel.appendChild(o2);
    }
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
    document.querySelectorAll('#factor-checks input[type=checkbox]:checked').forEach(cb => out.push(cb.value));
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
    const body = {
        universe,
        symbols: universe === 'custom' ? getCustomSymbols() : null,
        start_date: document.getElementById('start-year-select').value,
        end_date: document.getElementById('end-year-select').value,
        active_factors: factors,
        capital: parseFloat(document.getElementById('capital-input').value) || 100000,
        leverage: parseFloat(document.getElementById('leverage-slider').value) || 1.0,
        use_mwu: document.getElementById('mwu-hidden').value === '1',
        use_vol_target: document.getElementById('vol-hidden').value === '1',
        vol_target_pct: parseInt(document.getElementById('vol-pct-slider').value) / 100,
        strategy_mode: document.getElementById('strategy-mode-select').value,
        top_n: parseInt(document.getElementById('top-n-select').value) || 30,
        neutralize_sector: document.getElementById('sector-hidden').value === '1',
        factor_weights: _presetWeights || null,
        strategy_preset: _strategyPreset || null,
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
    const universe = document.getElementById('universe-select').value;
    const body = {
        account: (_config && _config.accounts && _config.accounts[0]) || 'Main',
        strategy_preset: _strategyPreset || null,
        active_factors: factors,
        universe: universe === 'custom' ? 'spy_qqq' : universe,
        leverage: parseFloat(document.getElementById('leverage-slider').value) || 1.0,
        top_n: parseInt(document.getElementById('top-n-select').value) || 20,
        factor_weights: _presetWeights || null,
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
    const rows = [
        { l: 'FINAL EQUITY', v: '$' + fmtNum(m.final_equity, 0), c: 'pos' },
        { l: 'TOTAL RETURN', v: fmtNum(m.total_return_pct, 1) + '%', c: m.total_return_pct >= 0 ? 'pos' : 'neg' },
        { l: 'CAGR', v: fmtNum(m.cagr_pct, 1) + '%', c: m.cagr_pct >= 0 ? 'pos' : 'neg' },
        { l: 'SHARPE', v: fmtNum(m.sharpe, 2), c: m.sharpe >= 1 ? 'pos' : '' },
        { l: 'CALMAR', v: fmtNum(m.calmar, 2), c: '' },
        { l: 'MAX DD', v: fmtNum(m.max_dd_pct, 1) + '%', c: 'neg' },
        { l: 'WIN RATE', v: fmtNum(m.win_rate_pct, 1) + '%', c: '' },
        { l: 'DAYS', v: (m.total_days || 0).toLocaleString(), c: '' },
    ];
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
    rows.slice(-60).reverse().forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${r.date}</td><td style="color:var(--green);font-size:10px;">${r.long || '--'}</td><td style="color:var(--red);font-size:10px;">${r.short || '--'}</td>`;
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
