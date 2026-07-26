"""Offline, immutable runner for the Seed-30 LETF rotation study.

The runner consumes a verified Research-v2 snapshot and writes only beneath
``research_v2/runs``.  It never imports the production store, scheduler, OMS,
server, or live configuration.  The default universe is intentionally labelled
``current-survivor Seed-30``: daily eligibility is point-in-time, but the seed
itself is not a dead-fund-inclusive historical product master.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import uuid


MODULE_DIR = Path(__file__).parent
DEFAULT_CONFIG = MODULE_DIR / "configs" / "letf_rotation_seed30.json"
DEFAULT_SNAPSHOT = MODULE_DIR / "snapshots" / "letf-sip-clean-20260717-v2"
DEFAULT_RUNS = MODULE_DIR / "runs"
REPRODUCIBILITY_SOURCES = (
    MODULE_DIR / "safety.py",
    MODULE_DIR / "snapshot.py",
    MODULE_DIR / "letf_universe.py",
    MODULE_DIR / "letf_rotation.py",
    MODULE_DIR / "letf_experiment.py",
    MODULE_DIR / "run_letf_rotation.py",
    MODULE_DIR / "portfolio.py",
    MODULE_DIR / "backtest.py",
    MODULE_DIR / "costs.py",
    MODULE_DIR / "metrics.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-07-17")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    parser.add_argument("--full-robustness", action="store_true")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(stage: str, **details: Any) -> None:
    print(
        json.dumps({"event": "letf_rotation_progress", "stage": stage, **_safe(details)}),
        flush=True,
    )


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if type(value).__module__.startswith(("numpy", "pandas")) and hasattr(value, "item"):
        return _safe(value.item())
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(_safe(value), indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _safe(value), sort_keys=True, separators=(",", ":"), default=str, allow_nan=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> Mapping[str, str]:
    """Hash every source file that can change the formal LETF result."""

    return {
        path.relative_to(MODULE_DIR.parent).as_posix(): _file_hash(path)
        for path in REPRODUCIBILITY_SOURCES
    }


def _snapshot_marker_fields(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return only explicit fields supplied by verified snapshot metadata."""

    snapshot_id = metadata.get("snapshot_id")
    manifest = metadata.get("manifest")
    data_sha256 = metadata.get("snapshot_data_sha256")
    data_file_count = metadata.get("snapshot_data_file_count")
    hash_scheme = metadata.get("snapshot_data_hash_scheme")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("verified snapshot metadata is missing snapshot_id")
    if not isinstance(manifest, Mapping):
        raise ValueError("verified snapshot metadata is missing manifest identity")
    manifest_sha256 = manifest.get("sha256")
    for label, value in (
        ("snapshot manifest", manifest_sha256),
        ("snapshot data", data_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"verified {label} SHA-256 is missing or invalid")
    if (
        isinstance(data_file_count, bool)
        or not isinstance(data_file_count, int)
        or data_file_count < 1
    ):
        raise ValueError("verified snapshot data file count is missing or invalid")
    if not isinstance(hash_scheme, str) or not hash_scheme:
        raise ValueError("verified snapshot data hash scheme is missing")
    return {
        "snapshot_id": snapshot_id,
        "snapshot_manifest_sha256": manifest_sha256,
        "snapshot_data_sha256": data_sha256,
        "snapshot_data_file_count": data_file_count,
        "snapshot_data_hash_scheme": hash_scheme,
    }


def _atomic_write(path: Path, payload: bytes, root: Path) -> None:
    from .safety import ensure_research_output_path

    destination = ensure_research_output_path(path, research_root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = ensure_research_output_path(
        destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}",
        research_root=root,
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _normalise_dates(frame):
    import pandas as pd

    result = frame.copy()
    result["timestamp"] = (
        pd.to_datetime(result["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
    )
    result["symbol"] = result["symbol"].astype(str).str.upper()
    return result.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)


def _eligibility_matrix(bars, sessions, registry, warmup_sessions: int):
    """Vectorized equivalent of ``pit_eligible_instruments`` for every day."""

    import pandas as pd

    index = pd.DatetimeIndex(sessions)
    result = pd.DataFrame(False, index=index, columns=[item.ticker for item in registry])
    for instrument in registry:
        observed = set(
            bars.loc[bars["symbol"] == instrument.ticker, "timestamp"].tolist()
        )
        active = (index >= pd.Timestamp(instrument.valid_from))
        if instrument.valid_to is not None:
            active &= index <= pd.Timestamp(instrument.valid_to)
        present = pd.Series(
            [session in observed for session in index], index=index, dtype=int
        )
        # Rows that pre-date this security identity never help the warm-up,
        # even if a previous security reused its ticker.
        present.loc[~active] = 0
        contiguous = present.rolling(warmup_sessions, min_periods=warmup_sessions).sum().eq(
            warmup_sessions
        )
        result.loc[:, instrument.ticker] = active & contiguous.to_numpy(dtype=bool)
    return result


def _breadth_series(proxy_bars, sessions, registry, eligibility):
    import numpy as np
    import pandas as pd

    close = (
        proxy_bars.pivot(index="timestamp", columns="symbol", values="close")
        .reindex(pd.DatetimeIndex(sessions))
        .sort_index()
    )
    slow = close.rolling(200, min_periods=200).mean()
    positive_63 = close.div(close.shift(63)).gt(1.0)
    gate = close.gt(slow) & positive_63 & close.notna()
    rows = {}
    proxy_by_ticker = {item.ticker: item.proxy for item in registry}
    for session in sessions:
        eligible = [
            ticker for ticker in eligibility.columns if bool(eligibility.at[session, ticker])
        ]
        proxies = sorted({proxy_by_ticker[ticker] for ticker in eligible})
        if not proxies:
            rows[session] = np.nan
            continue
        values = [bool(gate.at[session, proxy]) if proxy in gate.columns else False for proxy in proxies]
        rows[session] = float(np.mean(values))
    return pd.Series(rows, dtype=float).sort_index()


def _rotation_members(registry):
    from .letf_rotation import LETFMember

    return tuple(
        LETFMember(
            symbol=item.ticker,
            proxy_symbol=item.proxy,
            theme=item.theme,
            macro_bucket=item.macro_bucket,
        )
        for item in registry
    )


def _activation_session(sessions, eligibility, registry, *, min_symbols=12, min_themes=6):
    themes = {item.ticker: item.theme for item in registry}
    for position, session in enumerate(sessions):
        if position < 200:
            continue
        symbols = [ticker for ticker in eligibility.columns if eligibility.at[session, ticker]]
        if len(symbols) >= min_symbols and len({themes[ticker] for ticker in symbols}) >= min_themes:
            return session
    raise ValueError("no session satisfies the minimum dynamic-universe activation gate")


def _build_signals(
    *,
    sessions,
    product_bars,
    proxy_bars,
    registry,
    eligibility,
    rotation_config,
    activation_session,
    capture_audit: bool,
):
    from .letf_rotation import evaluate_rotation, is_rebalance_session

    members = _rotation_members(registry)
    tickers = tuple(member.symbol for member in members)
    signals = {}
    audit_rows = []
    decision_rows = []
    held = ()
    for session in sessions:
        if session < activation_session or not is_rebalance_session(
            session, sessions, rotation_config
        ):
            continue
        eligibility_map = {
            ticker: bool(eligibility.at[session, ticker])
            for ticker in tickers
        }
        decision = evaluate_rotation(
            session=session,
            sessions=sessions,
            product_bars=product_bars,
            proxy_bars=proxy_bars,
            universe=members,
            eligibility=eligibility_map,
            held_symbols=held,
            config=rotation_config,
        )
        if not decision.rebalance_due:
            continue
        signals[session] = {
            symbol: float(decision.scores[symbol]) for symbol in decision.selected
        }
        held = decision.selected
        decision_rows.append(
            {
                "signal_session": session,
                "selected": list(decision.selected),
                "selected_count": len(decision.selected),
                "cash_slots": decision.cash_slots,
                "eligible_count": sum(eligibility_map.values()),
            }
        )
        if capture_audit:
            for item in decision.audits:
                audit_rows.append(
                    {
                        "signal_session": session,
                        "symbol": item.symbol,
                        "accepted": item.accepted,
                        "reason": item.reason,
                        "score": item.score,
                        "details": json.dumps(_safe(item.details), sort_keys=True),
                    }
                )
    if not signals:
        raise ValueError("rotation generated no signals")
    return signals, decision_rows, audit_rows


def _ledger_frame(scenario):
    import pandas as pd

    rows = []
    for row in scenario.ledger:
        rows.append(
            {
                "session": row.session,
                "signal_generated": row.signal_generated,
                "executed_signal_session": row.executed_signal_session,
                "starting_equity": row.starting_equity,
                "ending_equity": row.ending_equity,
                "net_return": row.ending_equity / row.starting_equity - 1.0,
                "gross_exposure": row.gross_exposure,
                "net_exposure": row.net_exposure,
                "gross_turnover": row.gross_turnover,
                "total_cost": row.total_cost,
                "positions": json.dumps(_safe(row.positions), sort_keys=True),
                "executed_target_weights": json.dumps(
                    _safe(row.executed_target_weights), sort_keys=True
                ),
            }
        )
    return pd.DataFrame(rows)


def _benchmark_config(initial_capital: float, extra_friction_bps: float):
    from .letf_experiment import LETFExecutionConfig

    return LETFExecutionConfig(
        initial_capital=initial_capital,
        top_k=30,
        weighting="equal",
        gross_target=1.0,
        single_name_cap=1.0,
        theme_cap=1.0,
        rank_buffer=0,
        no_trade_band=0.0,
        target_volatility=None,
        trend_filter=False,
        breadth_exit=None,
        breadth_enter=None,
        risk_off_multiplier=1.0,
        drawdown_steps=(),
        extra_friction_bps=extra_friction_bps,
    )


def _monthly_equal_weight_signals(sessions, eligibility, activation_session):
    import pandas as pd

    index = pd.DatetimeIndex([session for session in sessions if session >= activation_session])
    month_ends = pd.Series(index=index, data=index).groupby(index.to_period("M")).last().tolist()
    return {
        pd.Timestamp(session): {
            ticker: 1.0
            for ticker in eligibility.columns
            if bool(eligibility.at[pd.Timestamp(session), ticker])
        }
        for session in month_ends
    }


def _metrics_row(label: str, scenario, benchmark=None):
    metrics = scenario.metrics.to_dict()
    row = {"scenario": label, **metrics}
    if benchmark is not None:
        row.update(_paired_return_comparison(scenario, benchmark))
    return row


def _return_series_metrics(returns) -> Mapping[str, float | int]:
    """Minimal performance metrics for an evaluation-only return control."""

    import numpy as np
    values = np.asarray(returns, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("return control needs at least two finite returns above -100%")
    growth = np.cumprod(1.0 + values)
    total_return = float(growth[-1] - 1.0)
    volatility = float(np.std(values, ddof=1) * math.sqrt(252.0))
    daily_std = float(np.std(values, ddof=1))
    equity = np.concatenate(([1.0], growth))
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "periods": int(len(values)),
        "total_return": total_return,
        "cagr": float((1.0 + total_return) ** (252.0 / len(values)) - 1.0),
        "annualized_volatility": volatility,
        "sharpe": (
            float(np.mean(values) / daily_std * math.sqrt(252.0))
            if daily_std > 0
            else 0.0
        ),
        "max_drawdown": float(np.min(drawdown)),
    }


def _paired_return_comparison(strategy, benchmark) -> Mapping[str, float | int | str]:
    """Compare against an ex-post equal-realized-volatility SPY control."""

    import numpy as np

    from .letf_experiment import paired_realized_volatility_control

    aligned, control = paired_realized_volatility_control(strategy, benchmark)
    strategy_total = float(np.prod(1.0 + aligned["strategy"].to_numpy(dtype=float)) - 1.0)
    benchmark_total = float(
        np.prod(1.0 + aligned["benchmark_control"].to_numpy(dtype=float)) - 1.0
    )
    return {
        "paired_sessions": int(len(aligned)),
        "paired_strategy_total_return": strategy_total,
        "paired_spy_realized_vol_control_total_return": benchmark_total,
        "total_return_minus_spy_realized_vol_control": strategy_total - benchmark_total,
        "spy_realized_vol_control_return_scale": float(
            control["benchmark_return_scale"]
        ),
        "paired_strategy_annualized_volatility": float(
            control["strategy_annualized_volatility"]
        ),
        "paired_spy_unscaled_annualized_volatility": float(
            control["benchmark_unscaled_annualized_volatility"]
        ),
        "paired_spy_control_annualized_volatility": float(
            control["benchmark_control_annualized_volatility"]
        ),
        "comparison_basis": str(control["comparison_basis"]),
    }


def _run(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    from .letf_experiment import (
        LETFExecutionConfig,
        build_market_context,
        cost_sensitivity,
        load_snapshot_bars,
        load_verified_snapshot_metadata,
        paired_moving_block_bootstrap,
        paired_realized_volatility_control,
        run_buy_and_hold,
        run_scenario,
        scenario_audit_payload,
        temporal_fold_metrics,
    )
    from .letf_rotation import RotationConfig, prepare_close_panel
    from .letf_universe import SEED30_REGISTRY, SEED30_TICKERS, registry_records
    from .safety import RESEARCH_ROOT, ensure_research_output_path

    if args.bootstrap_repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    config_path = Path(args.config).expanduser().resolve(strict=True)
    base_document = json.loads(config_path.read_text(encoding="utf-8"))
    if not base_document.get("research_only") or base_document.get("live_config_impact") != "none":
        raise ValueError("LETF config must be explicitly research-only with no live impact")
    if tuple(base_document["universe"]["tickers"]) != SEED30_TICKERS:
        raise ValueError("config ticker order differs from frozen Seed-30 registry")
    config_sha256 = _file_hash(config_path)
    source_hashes = _source_hashes()
    code_sha256 = _canonical_hash(source_hashes)

    snapshot = Path(args.snapshot).expanduser().resolve(strict=True)
    proxy_symbols = tuple(dict.fromkeys(item.proxy for item in SEED30_REGISTRY))
    required_symbols = tuple(dict.fromkeys((*SEED30_TICKERS, *proxy_symbols, "SPY", "QQQ")))
    bars, snapshot_metadata = load_snapshot_bars(
        snapshot,
        required_symbols,
        start=args.start,
        end=args.end,
        required_provider="Alpaca Markets",
        required_feed="SIP",
        required_adjustment="ALL",
        require_retrieved_at_utc=True,
    )
    snapshot_marker_fields = _snapshot_marker_fields(snapshot_metadata)
    bars = _normalise_dates(bars)
    _progress("snapshot_loaded", rows=len(bars), symbols=len(required_symbols))
    spy_dates = pd.DatetimeIndex(
        bars.loc[bars["symbol"] == "SPY", "timestamp"].drop_duplicates().sort_values()
    )
    if len(spy_dates) < 1_260:
        raise ValueError("fewer than five years of SPY sessions in the requested snapshot range")
    bars = bars.loc[bars["timestamp"].isin(spy_dates)].reset_index(drop=True)
    product_bars = bars.loc[bars["symbol"].isin(SEED30_TICKERS)].copy()
    proxy_bars = bars.loc[bars["symbol"].isin(proxy_symbols)].copy()
    product_close_panel = prepare_close_panel(product_bars)
    proxy_close_panel = prepare_close_panel(proxy_bars)

    warmup = int(base_document["universe"]["warmup_sessions"])
    eligibility = _eligibility_matrix(product_bars, spy_dates, SEED30_REGISTRY, warmup)
    activation = _activation_session(spy_dates, eligibility, SEED30_REGISTRY)
    breadth = _breadth_series(proxy_bars, spy_dates, SEED30_REGISTRY, eligibility)

    rotation_values = dict(base_document["rotation_defaults"])
    rotation_values["signal_weights"] = tuple(rotation_values["signal_weights"])
    rotation_config = RotationConfig(**rotation_values)
    if rotation_config.top_k != int(base_document["portfolio"]["top_n_default"]):
        raise ValueError("rotation default top_k conflicts with portfolio top_n_default")
    if rotation_config.cadence_sessions not in {
        int(value) for value in base_document["portfolio"]["rebalance_days_grid"]
    }:
        raise ValueError("rotation default cadence is absent from the declared grid")
    signals, decisions, selection_audit = _build_signals(
        sessions=spy_dates,
        product_bars=product_close_panel,
        proxy_bars=proxy_close_panel,
        registry=SEED30_REGISTRY,
        eligibility=eligibility,
        rotation_config=rotation_config,
        activation_session=activation,
        capture_audit=True,
    )
    _progress("primary_signals_built", signals=len(signals), activation=str(activation.date()))

    themes = {item.ticker: item.theme for item in SEED30_REGISTRY}
    themes["QQQ"] = "Benchmark:QQQ"
    execution_values = dict(base_document["execution_defaults"])
    execution_values["drawdown_steps"] = tuple(
        tuple(float(item) for item in step)
        for step in execution_values["drawdown_steps"]
    )
    primary_config = LETFExecutionConfig(
        initial_capital=float(args.initial_capital),
        **execution_values,
    )
    expected_execution_values = {
        "top_k": rotation_config.top_k,
        "weighting": str(base_document["portfolio"]["weighting_default"]),
        "gross_target": float(base_document["portfolio"]["gross_target_default"]),
        "single_name_cap": float(base_document["portfolio"]["single_name_cap"]),
        "theme_cap": float(base_document["portfolio"]["theme_cap"]),
        "max_adv_participation": float(base_document["costs"]["max_adv_participation"]),
        "target_volatility": float(base_document["risk"]["portfolio_target_volatility"]),
        "risk_target_change_buffer": float(base_document["risk"]["target_change_buffer"]),
        "breadth_exit": float(base_document["risk"]["breadth_exit"]),
        "breadth_enter": float(base_document["risk"]["breadth_reentry"]),
        "risk_off_multiplier": float(base_document["risk"]["risk_off_multiplier"]),
        "commission_bps": float(base_document["costs"]["commission_bps"]),
        "extra_friction_bps": float(base_document["costs"]["fixed_slippage_bps"]),
    }
    for field, expected in expected_execution_values.items():
        if getattr(primary_config, field) != expected:
            raise ValueError(f"execution default {field} conflicts with descriptive config")
    context_bars = bars.loc[
        bars["symbol"].isin((*SEED30_TICKERS, "SPY", "QQQ"))
    ].copy()
    context = build_market_context(
        context_bars,
        candidate_symbols=(*SEED30_TICKERS, "QQQ"),
        themes=themes,
        benchmark_symbol="SPY",
        breadth=breadth.dropna().to_dict(),
        config=primary_config,
    )
    primary = run_scenario(context, signals, config=primary_config)
    _progress("primary_backtest_complete", cagr=primary.metrics.cagr, sharpe=primary.metrics.sharpe)
    first_signal = min(signals)
    base_friction_bps = primary_config.extra_friction_bps
    spy = run_buy_and_hold(
        context,
        "SPY",
        signal_session=first_signal,
        extra_friction_bps=base_friction_bps,
    )
    qqq = run_buy_and_hold(
        context,
        "QQQ",
        signal_session=first_signal,
        extra_friction_bps=base_friction_bps,
    )
    upro = run_buy_and_hold(
        context,
        "UPRO",
        signal_session=first_signal,
        extra_friction_bps=base_friction_bps,
    )
    mandate_target_vol_spy = run_buy_and_hold(
        context,
        "SPY",
        signal_session=first_signal,
        extra_friction_bps=base_friction_bps,
        target_volatility=primary_config.target_volatility,
    )
    equal_weight_signals = _monthly_equal_weight_signals(
        spy_dates, eligibility, activation
    )
    equal_weight = run_scenario(
        context,
        equal_weight_signals,
        config=_benchmark_config(args.initial_capital, base_friction_bps),
    )
    _progress("benchmarks_complete")

    primary_control_returns, primary_control_metadata = (
        paired_realized_volatility_control(primary, mandate_target_vol_spy)
    )
    primary_control_metrics = _return_series_metrics(
        primary_control_returns["benchmark_control"]
    )
    bootstrap = paired_moving_block_bootstrap(
        primary,
        mandate_target_vol_spy,
        block_length=21,
        repetitions=args.bootstrap_repetitions,
        seed=20260720,
        realized_volatility_control=True,
    )
    costs = cost_sensitivity(
        context,
        signals,
        primary_config,
        friction_bps=tuple(base_document["costs"]["cost_sensitivity_bps"]),
    )
    _progress("cost_sensitivity_complete", scenarios=len(costs))
    strategy_folds = temporal_fold_metrics(primary)
    folds = []
    for row in strategy_folds:
        year = int(row["year"])
        benchmark_fold_returns = primary_control_returns.loc[
            primary_control_returns.index.year == year, "benchmark_control"
        ]
        if benchmark_fold_returns.empty:
            continue
        benchmark_fold_total = float(
            np.prod(1.0 + benchmark_fold_returns.to_numpy(dtype=float)) - 1.0
        )
        folds.append(
            {
                **row,
                "spy_realized_vol_control_total_return": benchmark_fold_total,
                "total_return_excess": (
                    row["total_return"] - benchmark_fold_total
                ),
            }
        )
    fold_positive_fraction = (
        float(np.mean([row["total_return_excess"] > 0 for row in folds]))
        if folds
        else 0.0
    )

    benchmark_rows = [
        _metrics_row("rotation_primary", primary, mandate_target_vol_spy),
        _metrics_row("SPY_buy_hold", spy),
        _metrics_row("QQQ_buy_hold", qqq),
        _metrics_row("UPRO_buy_hold", upro),
        _metrics_row("SPY_15pct_target_vol_mandate", mandate_target_vol_spy),
        {
            "scenario": "SPY_ex_post_realized_vol_control_for_primary",
            **primary_control_metrics,
            **primary_control_metadata,
        },
        _metrics_row("PIT_seed30_equal_weight_monthly", equal_weight),
    ]

    offset_rows = []
    for offset in range(rotation_config.cadence_sessions):
        cfg = replace(rotation_config, rebalance_offset=offset)
        offset_signals, _, _ = _build_signals(
            sessions=spy_dates,
            product_bars=product_close_panel,
            proxy_bars=proxy_close_panel,
            registry=SEED30_REGISTRY,
            eligibility=eligibility,
            rotation_config=cfg,
            activation_session=activation,
            capture_audit=False,
        )
        scenario = run_scenario(context, offset_signals, config=primary_config)
        offset_rows.append(
            {
                "offset": offset,
                **scenario.metrics.to_dict(),
                **_paired_return_comparison(scenario, mandate_target_vol_spy),
            }
        )
        _progress("offset_complete", offset=offset)

    neighbor_specs = []
    neighbor_names = set()
    for specification in base_document["robustness"]["neighbors"]:
        label = str(specification["name"])
        if not label or label in neighbor_names:
            raise ValueError(f"invalid or duplicate robustness neighbor {label!r}")
        neighbor_names.add(label)
        rotation_overrides = dict(specification.get("rotation", {}))
        if "signal_weights" in rotation_overrides:
            rotation_overrides["signal_weights"] = tuple(
                rotation_overrides["signal_weights"]
            )
        execution_overrides = dict(specification.get("execution", {}))
        if "drawdown_steps" in execution_overrides:
            execution_overrides["drawdown_steps"] = tuple(
                tuple(float(item) for item in step)
                for step in execution_overrides["drawdown_steps"]
            )
        signal_config = replace(rotation_config, **rotation_overrides)
        execution_config = replace(primary_config, **execution_overrides)
        if signal_config.top_k != execution_config.top_k:
            raise ValueError(
                f"neighbor {label!r} has inconsistent signal/execution top_k"
            )
        neighbor_specs.append((label, signal_config, execution_config))

    declared_top_n = {int(value) for value in base_document["portfolio"]["top_n_grid"]}
    tested_top_n = {rotation_config.top_k} | {
        signal_config.top_k for _, signal_config, _ in neighbor_specs
    }
    if not declared_top_n.issubset(tested_top_n):
        raise ValueError("robustness neighbors do not cover the declared top_n grid")
    declared_cadence = {
        int(value) for value in base_document["portfolio"]["rebalance_days_grid"]
    }
    tested_cadence = {rotation_config.cadence_sessions} | {
        signal_config.cadence_sessions for _, signal_config, _ in neighbor_specs
    }
    if not declared_cadence.issubset(tested_cadence):
        raise ValueError("robustness neighbors do not cover the declared cadence grid")
    grid_checks = {
        "weighting": (
            {str(value) for value in base_document["portfolio"]["weighting_grid"]},
            {primary_config.weighting}
            | {execution_config.weighting for _, _, execution_config in neighbor_specs},
        ),
        "gross_target": (
            {float(value) for value in base_document["portfolio"]["gross_target_grid"]},
            {primary_config.gross_target}
            | {execution_config.gross_target for _, _, execution_config in neighbor_specs},
        ),
        "rank_buffer": (
            {int(value) for value in base_document["portfolio"]["rank_buffer_grid"]},
            {primary_config.rank_buffer}
            | {execution_config.rank_buffer for _, _, execution_config in neighbor_specs},
        ),
        "no_trade_band": (
            {float(value) for value in base_document["portfolio"]["no_trade_band_grid"]},
            {primary_config.no_trade_band}
            | {execution_config.no_trade_band for _, _, execution_config in neighbor_specs},
        ),
    }
    for name, (declared, tested) in grid_checks.items():
        if not declared.issubset(tested):
            raise ValueError(
                f"robustness neighbors do not cover the declared {name} grid"
            )
    neighbor_rows = []
    for label, signal_config, execution_config in neighbor_specs:
        neighbor_signals, _, _ = _build_signals(
            sessions=spy_dates,
            product_bars=product_close_panel,
            proxy_bars=proxy_close_panel,
            registry=SEED30_REGISTRY,
            eligibility=eligibility,
            rotation_config=signal_config,
            activation_session=activation,
            capture_audit=False,
        )
        scenario = run_scenario(context, neighbor_signals, config=execution_config)
        neighbor_rows.append(
            {
                "neighbor": label,
                **scenario.metrics.to_dict(),
                **_paired_return_comparison(scenario, mandate_target_vol_spy),
            }
        )
        _progress("neighbor_complete", neighbor=label)

    # Diagnostics that deliberately move closer to the screenshot hypothesis:
    # rank the leveraged products themselves, then remove risk/group layers.
    # They are reported separately and never participate in champion selection.
    native_registry = tuple(replace(item, proxy=item.ticker) for item in SEED30_REGISTRY)
    native_signals, _, _ = _build_signals(
        sessions=spy_dates,
        product_bars=product_close_panel,
        proxy_bars=product_close_panel,
        registry=native_registry,
        eligibility=eligibility,
        rotation_config=rotation_config,
        activation_session=activation,
        capture_audit=False,
    )
    no_overlay_equal = LETFExecutionConfig(
        initial_capital=float(args.initial_capital),
        top_k=5,
        weighting="equal",
        gross_target=1.0,
        single_name_cap=0.20,
        theme_cap=0.20,
        rank_buffer=0,
        no_trade_band=0.0,
        max_adv_participation=primary_config.max_adv_participation,
        target_volatility=None,
        trend_filter=False,
        breadth_exit=None,
        breadth_enter=None,
        risk_off_multiplier=1.0,
        drawdown_steps=(),
        extra_friction_bps=base_friction_bps,
    )
    native_constrained = run_scenario(
        context, native_signals, config=no_overlay_equal
    )
    screenshot_rotation = replace(
        rotation_config,
        max_abs_correlation=1.0,
        max_per_theme=30,
        max_per_macro=30,
    )
    screenshot_signals, _, _ = _build_signals(
        sessions=spy_dates,
        product_bars=product_close_panel,
        proxy_bars=product_close_panel,
        registry=native_registry,
        eligibility=eligibility,
        rotation_config=screenshot_rotation,
        activation_session=activation,
        capture_audit=False,
    )
    screenshot_execution = replace(no_overlay_equal, theme_cap=1.0)
    screenshot_like = run_scenario(
        context, screenshot_signals, config=screenshot_execution
    )
    diagnostic_ablations = [
        _metrics_row(
            "product_native_group_corr_equal_no_risk",
            native_constrained,
            mandate_target_vol_spy,
        ),
        _metrics_row(
            "screenshot_like_product_top5_equal_no_risk",
            screenshot_like,
            mandate_target_vol_spy,
        ),
    ]
    _progress("diagnostic_ablations_complete")

    leave_symbol_rows = []
    leave_theme_rows = []
    if args.full_robustness:
        for excluded in SEED30_TICKERS:
            subset = tuple(item for item in SEED30_REGISTRY if item.ticker != excluded)
            subset_signals, _, _ = _build_signals(
                sessions=spy_dates,
                product_bars=product_close_panel,
                proxy_bars=proxy_close_panel,
                registry=subset,
                eligibility=eligibility,
                rotation_config=rotation_config,
                activation_session=activation,
                capture_audit=False,
            )
            scenario = run_scenario(context, subset_signals, config=primary_config)
            leave_symbol_rows.append(
                {
                    "excluded_symbol": excluded,
                    **scenario.metrics.to_dict(),
                    **_paired_return_comparison(scenario, mandate_target_vol_spy),
                }
            )
            _progress("leave_one_symbol_complete", excluded=excluded)
        for excluded_theme in sorted({item.theme for item in SEED30_REGISTRY}):
            subset = tuple(
                item for item in SEED30_REGISTRY if item.theme != excluded_theme
            )
            subset_signals, _, _ = _build_signals(
                sessions=spy_dates,
                product_bars=product_close_panel,
                proxy_bars=proxy_close_panel,
                registry=subset,
                eligibility=eligibility,
                rotation_config=rotation_config,
                activation_session=activation,
                capture_audit=False,
            )
            scenario = run_scenario(context, subset_signals, config=primary_config)
            leave_theme_rows.append(
                {
                    "excluded_theme": excluded_theme,
                    **scenario.metrics.to_dict(),
                    **_paired_return_comparison(scenario, mandate_target_vol_spy),
                }
            )
            _progress("leave_one_theme_complete", excluded=excluded_theme)

    neighbor_positive = float(
        np.mean([
            row["total_return_minus_spy_realized_vol_control"] > 0
            for row in neighbor_rows
        ])
    )
    cost_20 = next(
        row for row in costs if math.isclose(float(row["extra_friction_bps"]), 20.0)
    )
    primary_audit = scenario_audit_payload(primary)
    acceptance = base_document["acceptance"]
    worst_fold_excess = min(
        (float(row["total_return_excess"]) for row in folds), default=-math.inf
    )
    worst_fold_drawdown = min(
        (float(row["max_drawdown"]) for row in folds), default=-math.inf
    )
    symbol_loo_positive_fraction = (
        float(np.mean([
            row["total_return_minus_spy_realized_vol_control"] > 0
            for row in leave_symbol_rows
        ]))
        if leave_symbol_rows
        else 0.0
    )
    theme_loo_positive_fraction = (
        float(np.mean([
            row["total_return_minus_spy_realized_vol_control"] > 0
            for row in leave_theme_rows
        ]))
        if leave_theme_rows
        else 0.0
    )
    loo_required = bool(acceptance["leave_one_symbol_and_theme_out_required"])
    empirical_gates = {
        "overall_max_drawdown": primary.metrics.max_drawdown
        >= float(acceptance["overall_max_drawdown_limit"]),
        "net_sharpe": primary.metrics.sharpe
        >= float(acceptance["minimum_net_sharpe"]),
        "20bps_stress_sharpe": float(cost_20["metrics"]["sharpe"])
        >= float(acceptance["minimum_20bps_stress_sharpe"]),
        "positive_calendar_excess_fold_fraction": fold_positive_fraction
        >= float(acceptance["minimum_positive_calendar_excess_fold_fraction"]),
        "positive_worst_calendar_excess_fold": (
            not bool(acceptance["require_positive_worst_fold"])
            or worst_fold_excess > 0
        ),
        "worst_fold_max_drawdown": worst_fold_drawdown
        >= float(acceptance["worst_fold_max_drawdown_limit"]),
        "positive_neighbor_fraction": neighbor_positive
        >= float(acceptance["minimum_positive_neighbor_fraction"]),
        "bootstrap_outperformance_probability": float(bootstrap["probability_outperform"])
        >= float(acceptance["minimum_bootstrap_outperformance_probability"]),
        "all_offset_excess_positive": all(
            row["total_return_minus_spy_realized_vol_control"] > 0
            for row in offset_rows
        ),
        "leave_one_symbol_positive_fraction": (
            not loo_required
            or symbol_loo_positive_fraction
            >= float(acceptance["minimum_positive_leave_one_out_fraction"])
        ),
        "leave_one_theme_positive_fraction": (
            not loo_required
            or theme_loo_positive_fraction
            >= float(acceptance["minimum_positive_leave_one_out_fraction"])
        ),
        "no_executed_outer_gross_above_one": (
            int(primary_audit["executions_above_one_gross"]) == 0
        ),
        "adv_participation_within_one_percent": (
            primary.metrics.max_adv_participation
            <= primary_config.max_adv_participation + 1e-9
        ),
    }
    expected_theme_count = len({item.theme for item in SEED30_REGISTRY})
    structural_gates = {
        "snapshot_provenance_verified": True,
        "ticker_identity_regimes_verified": True,
        "dead_fund_inclusive_historical_master": False,
        "current_survivor_seed_bias_removed": False,
        "purged_walk_forward_completed": False,
        "untouched_lockbox_opened_once": False,
        "deflated_sharpe_and_pbo_completed": False,
        "leave_one_symbol_out_completed": (
            not loo_required or len(leave_symbol_rows) == len(SEED30_TICKERS)
        ),
        "leave_one_theme_out_completed": (
            not loo_required or len(leave_theme_rows) == expected_theme_count
        ),
        "configuration_neighborhood_completed": (
            not bool(acceptance["configuration_neighborhood_required"])
            or len(neighbor_rows) == len(neighbor_specs)
        ),
        "rebalance_offset_sensitivity_completed": (
            not bool(acceptance["rebalance_offset_sensitivity_required"])
            or len(offset_rows) == rotation_config.cadence_sessions
        ),
        "moving_block_bootstrap_completed": (
            not bool(acceptance["moving_block_bootstrap_required"])
            or int(bootstrap["repetitions"]) == int(args.bootstrap_repetitions)
        ),
        "three_month_shadow_or_paper_alignment": False,
        "live_config_unchanged": True,
    }

    provenance = snapshot_metadata["provenance"]
    resolved_data_audit = {
        "provider": provenance["provider"],
        "feed": provenance["feed"],
        "adjustment": provenance["adjustment"],
        "retrieved_at_utc": provenance["retrieved_at_utc"],
        "requested_evaluation_end": args.end,
        "snapshot_id": snapshot_metadata["snapshot_id"],
        "source_manifest_sha256": snapshot_metadata["manifest"]["sha256"],
        "snapshot_file_count": snapshot_metadata["file_count"],
        "snapshot_data_sha256": snapshot_metadata["snapshot_data_sha256"],
        "snapshot_data_hash_scheme": snapshot_metadata[
            "snapshot_data_hash_scheme"
        ],
        "held_missing_bar_policy": "fail_closed",
        "same_ticker_identity_policy": (
            "instrument identity begins 2025-02-20; tradable alias is FNGB through "
            "2025-06-23 and FNGU from 2025-06-24"
        ),
        "seed_survivorship_complete": False,
    }
    summary = {
        "format_version": 1,
        "created_at_utc": _utc_now(),
        "research_only": True,
        "live_config_impact": "none",
        "classification": "PIT_APPROX_INVALID_FOR_ALPHA_CLAIM",
        "classification_reason": (
            "Signals, eligibility, and execution are point-in-time, but the frozen Seed-30 "
            "contains current survivors and omits historically closed products. Structural "
            "validation therefore caps the result below deployable alpha regardless of return."
        ),
        "data_audit": resolved_data_audit,
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "source_sha256": source_hashes,
        "registry_sha256": _canonical_hash(registry_records()),
        "rotation_config": asdict(rotation_config),
        "execution_config": asdict(primary_config),
        "activation_session": str(activation.date()),
        "primary": primary_audit,
        "benchmarks": benchmark_rows,
        "paired_bootstrap_vs_spy_realized_vol_control": bootstrap,
        "cost_sensitivity": costs,
        "calendar_folds": folds,
        "offset_sensitivity": offset_rows,
        "parameter_and_layer_neighbors": neighbor_rows,
        "diagnostic_ablations_not_eligible_for_selection": diagnostic_ablations,
        "leave_one_symbol_out": leave_symbol_rows,
        "leave_one_theme_out": leave_theme_rows,
        "robustness_diagnostics": {
            "positive_calendar_excess_fold_fraction": fold_positive_fraction,
            "worst_calendar_excess_fold": worst_fold_excess,
            "worst_fold_max_drawdown": worst_fold_drawdown,
            "positive_neighbor_fraction": neighbor_positive,
            "positive_leave_one_symbol_out_fraction": symbol_loo_positive_fraction,
            "positive_leave_one_theme_out_fraction": theme_loo_positive_fraction,
        },
        "acceptance_thresholds": acceptance,
        "empirical_gates": empirical_gates,
        "structural_gates": structural_gates,
        "empirical_gate_pass_fraction": float(np.mean(list(empirical_gates.values()))),
        "structural_gate_pass_fraction": float(
            np.mean(list(structural_gates.values()))
        ),
    }

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "letf-seed30-%Y%m%dT%H%M%SZ"
    )
    root = Path(RESEARCH_ROOT).resolve()
    run_dir = ensure_research_output_path(DEFAULT_RUNS / run_id, research_root=root)
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    output_paths = []
    summary_path = run_dir / "summary.json"
    _atomic_write(summary_path, _json_bytes(summary), root)
    output_paths.append(summary_path)
    ledger_path = run_dir / "daily_ledger.csv"
    _atomic_write(ledger_path, _ledger_frame(primary).to_csv(index=False).encode("utf-8"), root)
    output_paths.append(ledger_path)
    decisions_path = run_dir / "rotation_decisions.csv"
    _atomic_write(
        decisions_path,
        pd.DataFrame(decisions).to_csv(index=False).encode("utf-8"),
        root,
    )
    output_paths.append(decisions_path)
    audit_path = run_dir / "selection_audit.csv"
    _atomic_write(
        audit_path,
        pd.DataFrame(selection_audit).to_csv(index=False).encode("utf-8"),
        root,
    )
    output_paths.append(audit_path)
    eligibility_path = run_dir / "eligibility.csv"
    eligibility_output = eligibility.copy()
    eligibility_output.insert(0, "session", eligibility_output.index)
    _atomic_write(
        eligibility_path,
        eligibility_output.to_csv(index=False).encode("utf-8"),
        root,
    )
    output_paths.append(eligibility_path)

    report_lines = [
        "# LETF Seed-30 Rotation Research",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Snapshot: `{resolved_data_audit['snapshot_id']}` (SIP / Adjustment.ALL)",
        f"- Evaluation: {primary.metrics.periods} sessions from {primary.ledger[0].session} to {primary.ledger[-1].session}",
        f"- Net CAGR / Sharpe / MaxDD: {primary.metrics.cagr:.2%} / {primary.metrics.sharpe:.3f} / {primary.metrics.max_drawdown:.2%}",
        (
            "- SPY ex-post realized-vol control CAGR / Sharpe / MaxDD: "
            f"{primary_control_metrics['cagr']:.2%} / "
            f"{primary_control_metrics['sharpe']:.3f} / "
            f"{primary_control_metrics['max_drawdown']:.2%} "
            f"(SPY return scale {primary_control_metadata['benchmark_return_scale']:.4f}; "
            "evaluation-only)"
        ),
        (
            "- SPY 15% target-vol mandate CAGR / Sharpe / MaxDD: "
            f"{mandate_target_vol_spy.metrics.cagr:.2%} / "
            f"{mandate_target_vol_spy.metrics.sharpe:.3f} / "
            f"{mandate_target_vol_spy.metrics.max_drawdown:.2%}"
        ),
        f"- Screenshot-like product Top-5 equal/no-risk CAGR / Sharpe / MaxDD: {screenshot_like.metrics.cagr:.2%} / {screenshot_like.metrics.sharpe:.3f} / {screenshot_like.metrics.max_drawdown:.2%}",
        f"- Paired block-bootstrap P(outperform): {float(bootstrap['probability_outperform']):.2%}",
        f"- Empirical gate pass fraction: {summary['empirical_gate_pass_fraction']:.2%}",
        "- Structural gate: FAIL — current-survivor seed, no dead-fund master, no untouched lockbox, no 3-month shadow alignment.",
        "",
        "The result is a research diagnostic, not authorization to change the live universe or use broker margin.",
    ]
    report_path = run_dir / "REPORT.md"
    _atomic_write(report_path, ("\n".join(report_lines) + "\n").encode("utf-8"), root)
    output_paths.append(report_path)

    completion_snapshot_metadata = load_verified_snapshot_metadata(
        snapshot,
        research_root=snapshot.parents[1],
        required_provider="Alpaca Markets",
        required_feed="SIP",
        required_adjustment="ALL",
        require_retrieved_at_utc=True,
    )
    if _snapshot_marker_fields(completion_snapshot_metadata) != snapshot_marker_fields:
        raise RuntimeError(
            "LETF research snapshot changed during the run; completion marker withheld"
        )
    if _source_hashes() != source_hashes:
        raise RuntimeError(
            "LETF research source changed during the run; completion marker withheld"
        )
    success = {
        "format_version": 1,
        "completed_at_utc": _utc_now(),
        "run_id": run_id,
        **snapshot_marker_fields,
        "config_sha256": config_sha256,
        "registry_sha256": summary["registry_sha256"],
        "code_sha256": code_sha256,
        "source_sha256": source_hashes,
        "outputs": {
            path.name: {"size": path.stat().st_size, "sha256": _file_hash(path)}
            for path in output_paths
        },
    }
    _atomic_write(run_dir / "_SUCCESS.json", _json_bytes(success), root)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary["primary"]["metrics"], "classification": summary["classification"]}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Heavy research imports happen only after broker credentials are removed
    # and production/live modules are blocked for this process.
    from .safety import offline_context

    with offline_context():
        return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
