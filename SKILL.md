# SKILL.md — ancserAPX development priorities

> Read this before touching portfolio / factor / execution code.

ancserAPX is a backtest-driven factor trading app over an Alpaca backend. The
single most important architectural rule in this repo is **backtest/live
parity**. Everything below exists to protect it.

---

## ⚠️ #1 PRIORITY: backtest and live MUST be identical

A backtest is only worth running if the live account makes the **exact same
decisions** the backtest made for the same preset and the same data. If the two
ever diverge, every backtest number becomes a lie and the whole app is
worthless. Treat any drift between them as a P0 bug.

### How parity is guaranteed: one shared source of truth

All portfolio-construction math lives in **`backend/alpha/portfolio.py`**. It is
a set of **pure functions** (no I/O, no Alpaca, no store, no globals): given the
factor Series + price Series + prior winner-lock state for ONE rebalance date,
they return target weights. That's it.

```
                       backend/alpha/portfolio.py
                       (composite_score, core_sleeve_weights,
                        satellite_weights, combined_target_weights)
                                    ▲                 ▲
                                    │                 │
        backend/backtest/engine.py │                 │ backend/execution/strategy.py
        run_strategy()  ───────────┘                 └───────────  LiveStrategy.calculate_targets()
        (feeds historical factor                       (feeds latest factor
         Series, one date per                           Series + persisted
         rebalance)                                     winner-lock state)
```

Both callers feed the **same function** the same shape of inputs, so they return
the same weights **by construction** — not by coincidence, not by two
implementations that "should" match.

### The rules

1. **NEVER re-implement weighting, ranking, sleeve-blending, or winner-lock
   logic anywhere except `backend/alpha/portfolio.py`.** Not in the engine, not
   in the live strategy, not in the server, not in a notebook helper. If you
   need that logic, import it.
2. **If you change a rule in `portfolio.py`, backtest and live both change
   together.** That is the entire point. Do not "temporarily" fork it.
3. **Keep `portfolio.py` pure.** No network, no file reads, no `datetime.now()`,
   no global state. Determinism is what makes parity testable.
4. **State is passed in and returned out.** Winner-lock state
   (`{sleeve: {symbol: {entry, locked}}}`) is an argument and a return value of
   `satellite_weights` / `combined_target_weights`. The backtest threads it
   through the simulation loop in memory; live persists it to
   `logs/winner_lock_state_{account}.json` between rebalances. Same logic, just
   different storage of the same state object.
5. **Factor inputs must come from the same code too.** Both callers use
   `backend/alpha/factors.py :: compute_all_factors` and the same
   `FACTOR_META` column/direction map. Don't compute a factor a second way.

### Sanity check before merging any change here

- Run the parity test (`scripts/verify_parity.py` if present, else the manual
  check below). Backtest rebalance-day weights for a preset must equal
  `LiveStrategy.calculate_targets` weights given the same as-of data and state.
- If you added a factor: it appears in `FACTOR_META` with the correct
  `descending` flag, and both backtest and live pick it up automatically because
  they read `FACTOR_META`.
- If you added a strategy knob (new sleeve option, new lock rule): it flows
  through `combined_target_weights` and therefore reaches both sides at once.

---

## Module map (parity-relevant)

| File | Role | Parity note |
|------|------|-------------|
| `backend/alpha/portfolio.py` | **Single source of truth** for target weights | Pure. Edit rules here only. |
| `backend/alpha/factors.py` | Factor computation + `FACTOR_META` + presets | Single source for factor math. |
| `backend/backtest/engine.py` | `run_strategy()` — historical sim | Calls `combined_target_weights`. No forked sim. |
| `backend/execution/strategy.py` | `LiveStrategy.calculate_targets()` | Sleeve path calls `combined_target_weights`; persists lock state. |
| `backend/execution/oms.py` | Turns target weights into Alpaca orders | Execution only — no portfolio math. |
| `backend/execution/scheduler.py` | Weekly rebalance gate | Controls *when*, not *what*. |
| `frontend/server.py` `/live/apply` | Writes the selected preset into the live config | Must write `sleeves` + `winner_lock` + `leverage` so live takes the sleeve path. |

---

## Strategy presets

`STRATEGY_PRESETS` in `backend/alpha/factors.py` defines full strategies
(sleeves + leverage + winner-lock), e.g. the current **Claude #1** baseline
(one long-only sleeve: 70% Momentum + 30% Reversion, no winner-lock, 1.5x,
weekly, top20). The UI
applies them via the **APPLY LIVE** button → `/live/apply`, which logs the exact
params and that they take effect on the **next** daily/weekly rebalance.

When you add a preset, you get parity for free *as long as* it only uses knobs
that flow through `combined_target_weights`. If a preset needs a brand-new
mechanic, add that mechanic to `portfolio.py` first.
