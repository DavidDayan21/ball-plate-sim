"""Comparison of the brains on the attempts of a session (app/comparison.py): pure computation."""
import pytest

from app.comparison import compare, conditions


def row(brain, action="throw", result="stabilized", stabilized=1.0, margin=50.0, setpoint="center",
        ball="ping_pong", friction=1.0, latency=20) -> dict:
    return dict(brain=brain, action=action, result=result, setpoint=setpoint, ball=ball, friction=friction,
                latency_ms=latency, stabilized_in_s=stabilized if result == "stabilized" else None,
                min_edge_margin_mm=margin)


def test_compare():
    rows = [row("pid", stabilized=1.0, margin=40.0), row("pid", stabilized=2.0, margin=60.0),
            row("pid", result="fallen", margin=-20.0), row("pid", result="not stabilized", margin=80.0),
            row("imitation", stabilized=1.5),
            row("pid", action="center"), row("pid", result="interrupted"),        # not counted
            row("imitation", ball="solid"), row("imitation", latency=60)]         # other conditions
    condition = conditions(rows[0])
    assert condition == ("center", "ping_pong", 1.0, 20)
    result = compare(rows, condition, ("pid", "imitation"))
    assert result["pid"] == dict(attempts=4, caught_pct=75.0, stabilized_s=1.5, margin_mm=60.0)
    assert result["imitation"] == dict(attempts=1, caught_pct=100.0, stabilized_s=1.5, margin_mm=50.0)


def test_compare_without_attempts():
    result = compare([row("pid", result="fallen")], ("center", "ping_pong", 1.0, 20), ("pid", "imitation"))
    assert result["pid"] == dict(attempts=1, caught_pct=0.0, stabilized_s=None, margin_mm=None)
    assert result["imitation"] == dict(attempts=0, caught_pct=None, stabilized_s=None, margin_mm=None)
    assert result["pid"]["caught_pct"] == pytest.approx(0)
