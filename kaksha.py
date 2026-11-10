"""KAKSHA core: event selection, the 2-day cutoff, event features and the competition metric.

Data: ESA Collision Avoidance Challenge (Zenodo 10.5281/zenodo.4463683, CC-BY-4.0).
Rules and metric: Uriot et al., "Spacecraft Collision Avoidance Challenge", Astrodynamics 2022
(arXiv 2008.03069): high risk means log10 Pc >= -6; only CDMs with time_to_tca >= 2 days may be
used; score L = MSE over high-risk events / F2; predictions may be clipped to -6.001.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent / "data"
HIGH = -6.0  # log10 Pc at or above this is high risk (competition threshold)
CLIP = -6.001  # predictions below HIGH are set here (allowed by the competition)
CUTOFF = 2.0  # days before closest approach; nothing closer may be used
FINAL = 1.0  # a usable event's last CDM must be within this many days of closest approach

TREND_COLS = ["risk", "miss_distance", "max_risk_estimate", "mahalanobis_distance",
              "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n"]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def split_cdms(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Training events -> (CDMs at least 2 days out, final risk per event).

    Keeps an event only if it has a CDM at or beyond the cutoff and its last CDM is within
    FINAL days of closest approach, as the competition did for its test set."""
    df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    last = df.groupby("event_id").tail(1).set_index("event_id")
    early = df[df.time_to_tca >= CUTOFF]
    keep = last.index[(last.time_to_tca < FINAL)].intersection(early.event_id.unique())
    return early[early.event_id.isin(keep)], last.loc[keep, "risk"].rename("final_risk")


def features(cdms: pd.DataFrame) -> pd.DataFrame:
    """One row per event from its CDMs. Refuses any CDM closer than the cutoff."""
    if (cdms.time_to_tca < CUTOFF).any():
        raise ValueError("CDM inside the 2-day cutoff passed to features()")
    cdms = cdms.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    g = cdms.groupby("event_id")
    last = g.tail(1).set_index("event_id")
    cols = {f"last_{c}": last[c] for c in last.columns if c not in ("mission_id", "c_object_type")}
    cols["mission_id"] = last.mission_id.astype("category")
    cols["object_type"] = last.c_object_type.astype("category")
    cols["n_cdms"] = g.size()
    cols["first_time_to_tca"] = g.time_to_tca.first()
    prev = g.nth(-2).set_index("event_id").reindex(last.index)  # NaN if the event has one CDM
    for c in TREND_COLS:
        cols[f"{c}_first"] = g[c].first()
        cols[f"{c}_max"] = g[c].max()
        cols[f"{c}_min"] = g[c].min()
        cols[f"{c}_std"] = g[c].std()
        cols[f"{c}_change"] = last[c] - prev[c]
    out = pd.DataFrame(cols)
    return out


def clip(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    return np.where(pred < HIGH, CLIP, pred)


def fbeta(true_high: np.ndarray, pred_high: np.ndarray, beta: float = 2.0) -> float:
    tp = np.sum(true_high & pred_high)
    fp = np.sum(~true_high & pred_high)
    fn = np.sum(true_high & ~pred_high)
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def score(true_risk, pred_risk) -> dict:
    """The competition's metrics plus the two an operator asks about."""
    t, p = np.asarray(true_risk, float), np.asarray(pred_risk, float)
    th, ph = t >= HIGH, p >= HIGH
    mse = float(np.mean((t[th] - p[th]) ** 2)) if th.any() else float("nan")
    f2 = fbeta(th, ph)
    return {
        "events": int(t.size), "high_risk_events": int(th.sum()),
        "mse_hr": mse, "f2": f2, "L": mse / f2 if f2 else float("inf"),
        **decision_counts(th, ph),
    }


def decision_counts(true_high, manoeuvre) -> dict:
    th, m = np.asarray(true_high, bool), np.asarray(manoeuvre, bool)
    return {
        "missed_high_risk": int(np.sum(th & ~m)),
        "recall_high_risk": float(np.sum(th & m) / max(th.sum(), 1)),
        "unnecessary_manoeuvres": int(np.sum(~th & m)),
        "false_alarm_rate": float(np.sum(~th & m) / max((~th).sum(), 1)),
    }
