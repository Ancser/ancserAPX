# ancserAPX

Alpaca stock trading research lab — factor backtest + live trading, dark fintech UI.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — fill APCA_API_KEY_ID and APCA_API_SECRET_KEY

# 3. Start the server
uvicorn frontend.server:app --host 0.0.0.0 --port 8080 --reload

# 4. Open in browser
# http://localhost:8080
```

On startup, the server automatically warms the broad stock universe plus QQQ/SPY
in the background. Every live run still performs its own authoritative broker-
eligibility check, incremental sync, 100% fresh-session check, and physical
Parquet/OHLCV validation before target calculation or orders.

To fetch the full 10-year history, click **FETCH 10Y** in the UI, or:

```bash
python -m backend.data.fetcher
```

## Daily scheduler

`ancserAPX install.bat` installs or updates the Windows task automatically. Its
trigger is calculated from **09:25 America/New_York** (five minutes before the
NYSE open), so a California host runs at 06:25. The launcher refreshes the local
trigger after each run for DST and rejects launches outside the 09:20–09:29 ET
pre-open safety window.

The installed task uses the current user's interactive logon token, so that user
must remain logged in. A second account-level execution lock prevents a website
click, daemon, and Windows task from mutating the same brokerage account at the
same time. If synchronization runs past the pre-open window, the scheduled run
is audited and blocked before OMS; an explicit manual Force can bypass only the
cadence/window, never the data or as-of gates.

```bash
# Daemon mode — remains running and checks at 09:25 ET
python -m backend.execution.scheduler

# One-shot (runs rebalance once and exits)
python -m backend.execution.scheduler --run-once

# Force rebalance even if already ran today
python -m backend.execution.scheduler --run-once --force
```

## API Keys Required

| Variable | Where to get |
|---|---|
| `APCA_API_KEY_ID` | alpaca.markets → Paper Trading → API Keys |
| `APCA_API_SECRET_KEY` | Same — shown only once at creation |

Paper trading keys are free. No credit card required for paper accounts.

## Features

| Feature | Status |
|---|---|
| Factor backtest (9 factors) | ✅ |
| SPY benchmark overlay | ✅ |
| MWU dynamic factor weights | ✅ |
| Vol targeting (leverage scaling) | ✅ |
| Sector neutralization | ✅ |
| Local Parquet data store | ✅ |
| Live dashboard (equity, positions) | ✅ |
| Force rebalance via UI | ✅ |
| Daily scheduler (APScheduler) | ✅ |
| Multi-account support | ✅ |
| WebSocket log streaming | ✅ |
| Pre-trade sync + freshness/as-of gate | ✅ |
| Durable order/fill/audit history | ✅ |
| Commission/slippage/regulatory cost model | ✅ |

## What's Missing / Limitations

- **10-year data depth**: Free IEX feed gives ~5 years. For full 10Y, set `APCA_DATA_FEED=sip` in `.env` and subscribe to Alpaca Algo Trader Plus (~$99/mo).
- **Live P&L scope**: daily equity change uses broker `equity - last_equity`.
  Final P&L in the UI is gross FIFO realized gain reconstructed from available
  fills. Fees, transfers, dividends, corporate actions and unmatched pre-history
  lots remain separate; the broker statement is authoritative.
- **Short selling**: Live short orders require a margin account. Backtest models shorts correctly.
- **Stop-loss / risk rules**: there is no per-position stop-loss. Portfolio risk
  controls include stateful 200EMA exit / 20EMA re-entry, risk-off leverage,
  daily volatility scaling, liquidity/shock filters and sector balancing.
- **Notifications**: No email/SMS alerts on rebalance. Logs stream to UI via WebSocket only.
- **Historical activities**: broker APIs can impose retention or pagination
  limits. The scheduler preserves every order/fill snapshot it observes plus an
  append-only local event stream; older unseen activity requires broker export.

## Multi-Account

Add additional accounts to `.env`:

```
APCA_API_KEY_ID_2=PK...
APCA_API_SECRET_KEY_2=...
APCA_PAPER_2=true
```

The UI will show all configured accounts in the account selector dropdown.
Each account can also override paper/live mode with `APCA_PAPER_<NAME>`
or legacy `PAPER_TRADING_<NAME>`. For example, `APCA_PAPER=false` on `Main`
and `APCA_PAPER_2=true` runs Main against the real Alpaca endpoint while
account `2` stays paper. Each account gets its own live config
(`config/live_strategy_2.json` for `_2`).
