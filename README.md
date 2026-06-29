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

On startup, the server automatically runs an incremental data sync in the background.

To fetch the full 10-year history, click **FETCH 10Y** in the UI, or:

```bash
python -m ancser.data.fetcher
```

## Running the Scheduler (optional)

```bash
# Daemon mode — rebalances at 09:35 ET daily
python -m ancser.execution.scheduler

# One-shot (runs rebalance once and exits)
python -m ancser.execution.scheduler --run-once

# Force rebalance even if already ran today
python -m ancser.execution.scheduler --run-once --force
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

## What's Missing / Limitations

- **10-year data depth**: Free IEX feed gives ~5 years. For full 10Y, set `APCA_DATA_FEED=sip` in `.env` and subscribe to Alpaca Algo Trader Plus (~$99/mo).
- **Live P&L tracking**: The tracker records daily state but doesn't compute intraday P&L — relies on Alpaca's portfolio history API.
- **Short selling**: Live short orders require a margin account. Backtest models shorts correctly.
- **Stop-loss / risk rules**: No per-position stop-loss. Risk management is via vol targeting only.
- **Notifications**: No email/SMS alerts on rebalance. Logs stream to UI via WebSocket only.
- **Historical activities**: `GET /live/activities` paginates up to 100 orders. Older history requires Alpaca's full export.

## Multi-Account

Add additional accounts to `.env`:

```
APCA_API_KEY_ID_2=PK...
APCA_API_SECRET_KEY_2=...
```

The UI will show all configured accounts in the account selector dropdown.
Each account gets its own live config (`config/live_strategy_Account2.json`).
