"""Command-line entry point for the opt-in swing-trading strategy profile."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

_ANALYST_ALIASES = {"sentiment": "social"}
_VALID_ANALYSTS = {"market", "social", "news", "fundamentals"}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _risk_percent(value: str) -> float:
    parsed = _positive_float(value)
    if parsed > 100:
        raise argparse.ArgumentTypeError("must be at most 100")
    return parsed


def _analysis_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD format") from exc
    return value


def _analyst_list(value: str) -> tuple[str, ...]:
    analysts = tuple(
        _ANALYST_ALIASES.get(item.strip().lower(), item.strip().lower())
        for item in value.split(",")
        if item.strip()
    )
    unknown = sorted(set(analysts) - _VALID_ANALYSTS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown analyst(s): {', '.join(unknown)}; choose from "
            "market, sentiment, news, fundamentals"
        )
    if not analysts:
        raise argparse.ArgumentTypeError("select at least one analyst")
    return analysts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradingagents-swing",
        description=(
            "Run TradingAgents with a bounded swing-trading horizon and explicit "
            "entry, stop, target, reward/risk, and account-risk rules."
        ),
    )
    parser.add_argument("ticker", help="Yahoo Finance ticker, e.g. RELIANCE.NS or AAPL")
    parser.add_argument(
        "--date", type=_analysis_date, default=date.today().isoformat(), help="YYYY-MM-DD"
    )
    parser.add_argument("--asset-type", choices=("stock", "crypto"), default="stock")
    parser.add_argument("--min-hold-days", type=_positive_int, default=2)
    parser.add_argument("--max-hold-days", type=_positive_int, default=15)
    parser.add_argument("--min-reward-risk", type=_positive_float, default=2.0)
    parser.add_argument(
        "--max-account-risk-pct",
        type=_risk_percent,
        default=0.5,
        help="Maximum planned loss as a percentage of account equity (default: 0.5)",
    )
    parser.add_argument(
        "--evaluation-days",
        type=_positive_int,
        default=5,
        help="Trading-day outcome window used by the decision log (default: 5)",
    )
    parser.add_argument(
        "--analysts",
        type=_analyst_list,
        default=_analyst_list("market,sentiment,news,fundamentals"),
        help="Comma-separated: market,sentiment,news,fundamentals",
    )
    parser.add_argument("--provider", help="Override TRADINGAGENTS_LLM_PROVIDER")
    parser.add_argument("--deep-model", help="Override the configured deep-thinking model")
    parser.add_argument("--quick-model", help="Override the configured quick-thinking model")
    parser.add_argument("--checkpoint", action="store_true", help="Enable checkpoint/resume")
    parser.add_argument("--debug", action="store_true", help="Stream intermediate agent output")
    return parser


def build_swing_config(args: argparse.Namespace) -> dict:
    if args.max_hold_days < args.min_hold_days:
        raise ValueError("--max-hold-days must be greater than or equal to --min-hold-days")

    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "strategy_mode": "swing",
            "swing_min_hold_days": args.min_hold_days,
            "swing_max_hold_days": args.max_hold_days,
            "min_reward_risk": args.min_reward_risk,
            "max_account_risk_pct": args.max_account_risk_pct,
            "evaluation_horizon_days": args.evaluation_days,
            "checkpoint_enabled": args.checkpoint,
        }
    )
    if args.provider:
        config["llm_provider"] = args.provider
    if args.deep_model:
        config["deep_think_llm"] = args.deep_model
    if args.quick_model:
        config["quick_think_llm"] = args.quick_model
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = build_swing_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    graph = TradingAgentsGraph(
        selected_analysts=args.analysts,
        debug=args.debug,
        config=config,
    )
    final_state, rating = graph.propagate(args.ticker, args.date, asset_type=args.asset_type)
    report_path = graph.save_reports(final_state, args.ticker)

    print(f"\nSwing rating: {rating}")
    print(final_state["final_trade_decision"])
    print(f"\nReports saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
