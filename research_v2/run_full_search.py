"""Run the frozen-score, net-cost Research v2 strategy search.

The script consumes completed tabular and checkpointed sequence OOS artifacts.
All hybrid formulas and search grids below are declared before any confirmation
rows are loaded.  The resulting champion is therefore selected on selection
OOS only and merely evaluated on the already-opened 2026 confirmation period.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from typing import Mapping


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs" / "20260710_full_v1"
DEFAULT_CACHE = ROOT / "cache" / "canonical_features_h5.parquet"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(path: Path, payload: Mapping[str, object]) -> None:
    record = {"utc": _utc_now(), **dict(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(record, sort_keys=True, default=str), flush=True)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args(argv)


def _merge(tabular_path: Path, sequence_path: Path):
    import pandas as pd

    tabular = pd.read_parquet(tabular_path)
    sequence = pd.read_parquet(sequence_path)
    keys = ["timestamp", "symbol"]
    if tabular.duplicated(keys).any() or sequence.duplicated(keys).any():
        raise ValueError("prediction artifacts contain duplicate timestamp/symbol keys")
    sequence_scores = [name for name in sequence if name.startswith("score_")]
    keep = keys + sequence_scores
    merged = tabular.merge(
        sequence.loc[:, keep],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_sequence_duplicate"),
    )
    duplicates = [name for name in merged if name.endswith("_sequence_duplicate")]
    if duplicates:
        raise ValueError(f"tabular/sequence score names overlap: {duplicates}")
    if len(merged) != len(tabular) or len(merged) != len(sequence):
        raise ValueError(
            "tabular and sequence OOS keys are not identical; refusing an inner-join bias"
        )
    return merged.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _add_fixed_hybrids(frame):
    """Add pre-declared cross-sectionally calibrated model averages."""

    from research_v2.models import cross_sectional_percentile_rank

    result = frame.copy()
    dates = result["timestamp"]

    def rank(name: str):
        if name not in result:
            raise ValueError(f"required ensemble score is missing: {name}")
        return cross_sectional_percentile_rank(result[name], dates, center=True)

    tabular = rank("score_ensemble")
    sequence = rank("score_sequence_locked")
    ridge = rank("score_ridge")
    gbdt = rank("score_gbdt")
    gru = rank("score_gru")
    transformer = rank("score_transformer")

    # These formulas are intentionally fixed before confirmation is read.
    result["score_hybrid_tab80_seq20"] = 0.80 * tabular + 0.20 * sequence
    result["score_hybrid_tab60_seq40"] = 0.60 * tabular + 0.40 * sequence
    result["score_hybrid_tab50_seq50"] = 0.50 * tabular + 0.50 * sequence
    result["score_ml_four_equal"] = 0.25 * (ridge + gbdt + gru + transformer)
    result["score_tabular_linear_tree_equal"] = 0.50 * (ridge + gbdt)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    from research_v2.safety import ensure_research_output_path, offline_context

    run_dir = ensure_research_output_path(args.run_dir, research_root=ROOT)
    cache = args.cache.expanduser().resolve(strict=True)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    sequence_dir = run_dir / "sequence_full60_all"
    required_sequence = (
        sequence_dir / "_SUCCESS",
        sequence_dir / "summary.json",
        sequence_dir / "selection" / "predictions.parquet",
        sequence_dir / "lockbox" / "predictions.parquet",
    )
    missing = [str(path) for path in required_sequence if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "definitive sequence training is not complete: " + ", ".join(missing)
        )

    output = ensure_research_output_path(run_dir / "search", research_root=ROOT)
    if output.exists():
        raise FileExistsError(
            f"search output already exists; inspect it instead of overwriting: {output}"
        )
    staging = ensure_research_output_path(
        run_dir / f".search.partial-{uuid.uuid4().hex}", research_root=ROOT
    )
    staging.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "search_progress.jsonl"

    try:
        with offline_context():
            import pandas as pd
            import polars as pl

            from research_v2.backtest import RiskConfig
            from research_v2.costs import CostConfig
            from research_v2.experiment import build_market_context, complete_case_symbols
            from research_v2.portfolio import PortfolioConfig
            from research_v2.reporting import generate_research_report
            from research_v2.search import (
                Cadence,
                SearchPolicy,
                run_staged_search,
                write_search_artifacts,
            )

            started = time.perf_counter()
            _log(log_path, {"event": "search_process_started", "pid": os.getpid()})
            selection = _add_fixed_hybrids(
                _merge(
                    run_dir / "tabular" / "selection_oos_predictions.parquet",
                    sequence_dir / "selection" / "predictions.parquet",
                )
            )
            # Confirmation is transformed by exactly the same already-declared
            # formulas and is not inspected by the staged selector.
            confirmation = _add_fixed_hybrids(
                _merge(
                    run_dir / "tabular" / "lockbox_predictions.parquet",
                    sequence_dir / "lockbox" / "predictions.parquet",
                )
            )
            score_columns = (
                "score_production_claude1",
                "score_production_claude3",
                "score_momentum_ensemble_v2",
                "score_momentum_lowvol_v2",
                "score_ridge",
                "score_gbdt",
                "score_ensemble",
                "score_gru",
                "score_transformer",
                "score_sequence_locked",
                "score_hybrid_tab80_seq20",
                "score_hybrid_tab60_seq40",
                "score_hybrid_tab50_seq50",
                "score_ml_four_equal",
                "score_tabular_linear_tree_equal",
            )
            if any(
                selection[name].isna().any() or confirmation[name].isna().any()
                for name in score_columns
            ):
                raise ValueError("a declared strategy score contains missing predictions")

            market_columns = [
                "timestamp",
                "symbol",
                "open",
                "close",
                "adv20_v2",
                "realized_vol_20d_v2",
                "range_pct_v2",
                "breadth_200d_v2",
            ]
            panel = pl.read_parquet(cache, columns=market_columns)
            symbols = complete_case_symbols(panel)
            start = min(selection["timestamp"].min(), confirmation["timestamp"].min())
            end = max(selection["timestamp"].max(), confirmation["timestamp"].max())
            market = build_market_context(
                panel,
                symbols=symbols,
                start=start,
                end=end,
                beta_lookback=126,
                spread_range_fraction=0.02,
                min_spread_bps=1.0,
                max_spread_bps=30.0,
            )

            portfolio = PortfolioConfig(
                top_n=20,
                weighting="inverse_vol",
                gross_target=1.0,
                single_name_cap=0.10,
                sector_cap=0.30,
                inverse_vol_floor=0.005,
                rank_buffer=5,
                no_trade_band=0.0025,
                staggered_tranches=1,
                max_adv_participation=0.02,
            )
            cost = CostConfig(
                commission_bps=0.0,
                spread_multiplier=1.0,
                min_spread_bps=0.0,
                max_spread_bps=30.0,
                impact_coefficient=0.10,
                max_impact_bps=50.0,
                max_adv_participation=0.02,
                annual_funding_rate=0.055,
                periods_per_year=252,
            )
            unmanaged = RiskConfig(target_change_buffer=0.0025)
            vol_dd_beta = RiskConfig(
                target_volatility=0.18,
                vol_lookback=63,
                min_vol_observations=30,
                drawdown_steps=((0.10, 0.67), (0.20, 0.33)),
                max_abs_beta=1.25,
                target_change_buffer=0.0025,
            )
            full_throttle = RiskConfig(
                target_volatility=0.18,
                vol_lookback=63,
                min_vol_observations=30,
                drawdown_steps=((0.10, 0.67), (0.20, 0.33)),
                max_abs_beta=1.25,
                trend_filter=True,
                breadth_exit=0.30,
                breadth_enter=0.45,
                risk_off_multiplier=0.50,
                crowding_threshold=0.65,
                crowding_multiplier=0.70,
                target_change_buffer=0.0025,
            )
            full_cash = RiskConfig(
                target_volatility=0.18,
                vol_lookback=63,
                min_vol_observations=30,
                drawdown_steps=((0.10, 0.67), (0.20, 0.33)),
                max_abs_beta=1.25,
                trend_filter=True,
                breadth_exit=0.30,
                breadth_enter=0.45,
                risk_off_multiplier=0.0,
                crowding_threshold=0.65,
                crowding_multiplier=0.70,
                target_change_buffer=0.0025,
            )
            risk_variants = {
                "none": unmanaged,
                "vol_dd_beta": vol_dd_beta,
                "full_throttle": full_throttle,
                "full_cash": full_cash,
            }
            cadences = (
                Cadence("weekly", 5, 1),
                Cadence("daily_5_tranches", 1, 5),
                Cadence("biweekly", 10, 1),
                Cadence("monthly", 21, 1),
            )
            _log(
                log_path,
                {
                    "event": "search_plan_locked",
                    "selection_rows": len(selection),
                    "confirmation_rows": len(confirmation),
                    "scores": list(score_columns),
                    "top_n": [15, 20, 30],
                    "cadences": [item.name for item in cadences],
                    "weighting": ["equal", "inverse_vol"],
                    "risk_variants": list(risk_variants),
                    "leverage": [0.75, 1.0, 1.25, 1.5],
                    "extra_cost_bps": [0.0, 5.0, 10.0, 20.0],
                    "selection_cost_bps": 10.0,
                },
            )

            def progress(event: Mapping[str, object]) -> None:
                _log(log_path, event)

            result = run_staged_search(
                market,
                selection,
                confirmation,
                score_columns=score_columns,
                base_portfolio=portfolio,
                base_cost=cost,
                base_risk=full_throttle,
                top_n_grid=(15, 20, 30),
                cadence_grid=cadences,
                weighting_grid=("equal", "inverse_vol"),
                risk_variants=risk_variants,
                leverage_grid=(0.75, 1.0, 1.25, 1.5),
                cost_sensitivity_bps=(0.0, 5.0, 10.0, 20.0),
                selection_cost_bps=10.0,
                base_rebalance_days=5,
                policy=SearchPolicy(
                    objective="sharpe",
                    require_positive_worst_fold=False,
                    max_drawdown_limit=-0.35,
                ),
                initial_capital=100_000.0,
                progress=progress,
            )
            write_search_artifacts(result, staging, research_root=ROOT)
            (staging / "_SUCCESS.json").write_text(
                json.dumps(
                    {
                        "completed_at_utc": _utc_now(),
                        "champion_candidate_id": result.champion.candidate.candidate_id,
                        "selection_only": True,
                        "confirmation_previously_opened": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(staging, output)
            paths = generate_research_report(run_dir, research_root=ROOT)
            _log(
                log_path,
                {
                    "event": "search_and_report_completed",
                    "champion_candidate_id": result.champion.candidate.candidate_id,
                    "elapsed_seconds": time.perf_counter() - started,
                    "report_json": str(paths.json),
                    "report_markdown": str(paths.markdown),
                },
            )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
