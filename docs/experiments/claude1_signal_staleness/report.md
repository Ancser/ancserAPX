# Claude #1 signal-staleness diagnostic

Generated from the local store through 2026-07-13. This
is a controlled sensitivity test: execution dates and cost assumptions are the
same, while the challenger always uses the factor snapshot from exactly five
trading sessions earlier.

| Variant | Final equity | CAGR | Sharpe | MaxDD | Gross turnover | Trading cost |
|---|---:|---:|---:|---:|---:|---:|
| Current signal | $653,565.00 | 45.48% | 1.18 | -31.95% | 363.08x | $45,690.43 |
| Signal delayed 5 sessions | $592,305.65 | 42.65% | 1.11 | -36.34% | 359.37x | $43,028.04 |

Average same-name overlap was 53.53%
of the normal Top-20 book (10.71 names),
with mean Jaccard overlap 37.51% across
252 matched rebalances.

Cost assumptions: 0 bps broker commission on buys/sells,
5 bps spread/slippage on buys/sells, and
0 bps blended regulatory fee on sells. Zero Alpaca
broker commission is not treated as zero execution cost.

The same current-signal strategy with all modeled friction set to zero ended at $782,928.85 with 50.82% CAGR. The default friction therefore changed final equity by $-129,363.85 and CAGR by -5.34 percentage points. Dollar fees and final-equity drag differ because foregone capital no longer compounds.

## Limits

- Static present-day constituent list creates survivorship bias.
- Daily close-to-close bars cannot reproduce open-auction fills, quoted spreads, partial fills, or TAF per-share caps.
- The production backtest's 5-session cadence is not a calendar-Friday scheduler around market holidays.
- The stale variant isolates signal age only; it does not simulate missing symbols, corporate-action corruption, or order failures.
- Sessions below 95% of the trailing 60-session median symbol count are excluded as partial incremental-sync snapshots.
