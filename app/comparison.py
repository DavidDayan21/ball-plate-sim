"""Comparison of the brains on the finished attempts of the session (rows of Session.summary). Without Qt.

Only the attempts made in the same conditions are compared: setpoint (the same drawn path, at the same speed), ball,
friction, latency. An attempt counts if
it tested the brain (a throw, a drop, a push, a disturbance or a new setpoint; not a centering nor a free attempt),
and if it has a result (not interrupted by another action or a change of setting).

Indicators, by brain:
    attempts       number of attempts counted
    caught_pct     share of them where the ball did not fall (%)
    stabilized_s   median time to stabilize, of those that stabilized (s)
    margin_mm      median of the closest approach to the edge of the plate, of those caught (mm): how much room
                   the brain kept (the largest distance to the setpoint, by contrast, mostly tells where the ball
                   was thrown from)
"""
import numpy as np

COUNTED = ("throw", "drop", "push", "disturbance", "setpoint")


def conditions(row: dict) -> tuple:
    """What an attempt was made in, besides its brain: (setpoint, drawn path, ball, friction, latency in ms)."""
    return row["setpoint"], row.get("path", ""), row["ball"], row["friction"], row["latency_ms"]


def compare(rows, condition: tuple, brains) -> dict:
    """{brain: {indicator: value}} over the rows made in `condition`; None where there is nothing to compute."""
    out = {}
    for brain in brains:
        mine = [r for r in rows if r["brain"] == brain and r["action"] in COUNTED and r["result"] != "interrupted"
                and conditions(r) == condition]
        caught = [r for r in mine if r["result"] != "fallen"]
        stabilized = [r["stabilized_in_s"] for r in mine if r["result"] == "stabilized"]
        margins = [r["min_edge_margin_mm"] for r in caught if r["min_edge_margin_mm"] is not None]
        out[brain] = dict(attempts=len(mine),
                          caught_pct=100 * len(caught) / len(mine) if mine else None,
                          stabilized_s=float(np.median(stabilized)) if stabilized else None,
                          margin_mm=float(np.median(margins)) if margins else None)
    return out
