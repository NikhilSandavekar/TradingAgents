"""Coverage for the opt-in swing-trading strategy profile."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cli.swing import build_parser, build_swing_config
from tradingagents.agents.schemas import (
    TraderAction,
    TraderProposal,
    render_trader_proposal,
)
from tradingagents.agents.utils.agent_utils import get_instrument_context_from_state
from tradingagents.dataflows.config import set_config
from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.unit
def test_general_mode_does_not_change_instrument_context():
    state = {"company_of_interest": "RELIANCE.NS", "asset_type": "stock"}
    context = get_instrument_context_from_state(state)
    assert "Swing-trading mandate" not in context


@pytest.mark.unit
def test_swing_mode_reaches_shared_agent_context():
    set_config(
        {
            "strategy_mode": "swing",
            "swing_min_hold_days": 3,
            "swing_max_hold_days": 12,
            "min_reward_risk": 2.5,
            "max_account_risk_pct": 0.75,
        }
    )
    state = {"company_of_interest": "RELIANCE.NS", "asset_type": "stock"}
    context = get_instrument_context_from_state(state)
    assert "3-12 trading-day opportunity" in context
    assert "2.5:1" in context
    assert "0.75% of account equity" in context
    assert "Use Hold" in context


@pytest.mark.unit
@pytest.mark.parametrize(
    "override,match",
    [
        ({"swing_min_hold_days": 0}, "holding period"),
        ({"swing_min_hold_days": 10, "swing_max_hold_days": 5}, "holding period"),
        ({"min_reward_risk": 0}, "min_reward_risk"),
        ({"max_account_risk_pct": 0}, "max_account_risk_pct"),
    ],
)
def test_invalid_swing_settings_fail_loudly(override, match):
    config = {"strategy_mode": "swing", **override}
    set_config(config)
    with pytest.raises(ValueError, match=match):
        get_instrument_context_from_state({"company_of_interest": "AAPL"})


@pytest.mark.unit
def test_trader_proposal_renders_complete_swing_setup():
    proposal = TraderProposal(
        action=TraderAction.BUY,
        reasoning="Breakout setup with ATR-defined risk.",
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        reward_risk_ratio=2.0,
        time_horizon="2-10 trading days",
        position_sizing="shares = floor(account equity * 0.005 / 5)",
    )
    rendered = render_trader_proposal(proposal)
    assert "**Target Price**: 110.0" in rendered
    assert "**Reward/Risk**: 2:1" in rendered
    assert "**Time Horizon**: 2-10 trading days" in rendered
    assert "**Position Sizing**" in rendered


@pytest.mark.unit
def test_swing_cli_builds_opt_in_config():
    args = build_parser().parse_args(
        [
            "RELIANCE.NS",
            "--min-hold-days",
            "3",
            "--max-hold-days",
            "12",
            "--min-reward-risk",
            "2.5",
            "--max-account-risk-pct",
            "0.75",
            "--analysts",
            "market,sentiment,news",
        ]
    )
    config = build_swing_config(args)
    assert config["strategy_mode"] == "swing"
    assert config["swing_min_hold_days"] == 3
    assert config["swing_max_hold_days"] == 12
    assert config["min_reward_risk"] == 2.5
    assert config["max_account_risk_pct"] == 0.75
    assert args.analysts == ("market", "social", "news")


@pytest.mark.unit
def test_memory_reflection_uses_configured_evaluation_window():
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"evaluation_horizon_days": 8}
    graph.memory_log = MagicMock()
    graph.memory_log.get_pending_entries.return_value = [
        {"ticker": "AAPL", "date": "2026-01-05", "decision": "Hold"}
    ]
    graph._resolve_benchmark = MagicMock(return_value="SPY")
    graph._fetch_returns = MagicMock(return_value=(None, None, None, None))

    TradingAgentsGraph._resolve_pending_entries(graph, "AAPL")

    graph._fetch_returns.assert_called_once_with(
        "AAPL", "2026-01-05", holding_days=8, benchmark="SPY"
    )
