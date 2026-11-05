"""Campaign tools: draws, statistics over the draws, bisection of the maximum speeds, full pipeline."""
from dataclasses import replace
import math

import pytest

import campaign as cp
from simulation.config import Config


def test_draw_zero_is_the_nominal_configuration():
    assert cp.draw(cp.BASE, 0) is cp.BASE


def test_draws_reproducible_bounded_and_distinct():
    cfg = Config()
    x0, y0 = cfg.start_position
    offsets = set()
    for k in range(1, 21):
        d = cp.draw(cfg, k)
        assert d.seed == k and d.start_position == cp.draw(cfg, k).start_position
        dx, dy = d.start_position[0] - x0, d.start_position[1] - y0
        assert abs(dx) <= cp.START_OFFSET and abs(dy) <= cp.START_OFFSET
        offsets.add((dx, dy))
    assert len(offsets) == 20


def test_mean_std_ignores_fallen_draws():
    m, s = cp.mean_std([1.0, 2.0, math.nan, 3.0])
    assert m == 2.0 and s == 1.0
    assert math.isnan(cp.mean_std([4.0])[1]) and math.isnan(cp.mean_std([math.nan])[0])


def test_paired_delta_significance():
    a = [1.0, 2.0, 3.0, 4.0]
    m, se, significant = cp.paired_delta(a, [1.1, 2.1, 3.2, 4.1])     # differences 0.1, 0.1, 0.2, 0.1
    assert m == pytest.approx(0.125) and se == pytest.approx(0.025) and significant   # 0.125 > 3.182 × 0.025
    assert not cp.paired_delta(a, [1.1, 1.9, 3.1, 3.9])[2]              # differences ±0.1: mean 0
    # Two draws only: t = 12.7, a mean of 3 standard errors is not significant (it would be with t = 2)
    assert not cp.paired_delta([0.0, 0.0], [0.2, 0.4])[2]


def test_max_caught_by_bisection():
    tested_speeds = []

    def caught(v):
        tested_speeds.append(v)
        return v <= 0.537

    v, tested = cp.max_caught(caught, 0.1, 1.5, 0.01)
    assert v == 0.53 and len(tested) <= 10 and [s for s, _ in tested] == tested_speeds
    assert cp.max_caught(lambda v: False, 0.1, 1.5, 0.01)[0] is None
    assert cp.max_caught(lambda v: True, 0.1, 1.5, 0.01)[0] == 1.5


def test_campaign_end_to_end(tmp_path, monkeypatch):
    # Whole pipeline (simulations, bisections, report, figures) in simple physics, to stay fast
    simple = {"physics": "simple", "mechanism": False, "sensor": "synthetic"}
    monkeypatch.setattr(cp, "BASE", replace(cp.BASE, **simple))
    monkeypatch.setattr(cp, "CASES", {name: replace(cfg, **simple) for name, cfg in cp.CASES.items()})
    monkeypatch.setattr(cp.plt, "show", lambda: None)
    data = cp.run_campaign(2, processes=1)
    assert all(len(runs) == 2 and runs[0].x != runs[1].x for runs in data["runs"].values())
    assert all(cp.monotonic(name, data) == "yes" for name in cp.LIMITS)
    report = tmp_path / "report.md"
    cp.write_report(data, str(report))
    text = report.read_text(encoding="utf-8")
    assert all(f"## {i}." in text for i in range(7)) and "| 0/2 |" in text and " ± " in text
    cp.show_figures(data)
    cp.plt.close("all")
