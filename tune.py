"""Model selection on ESA's TRAIN events only (the official test set is never read here).

Each row is one event, so stratified K-fold over rows is a split by event: no event's CDMs sit on
both sides. 5 folds x 5 repeats; every candidate is fitted on the training folds and scored on the
held-out fold with the competition metric (L = MSE on high-risk events / F2). Writes
results/validation.csv.

A candidate = a decision rule (which events are flagged high risk) + a value rule (what log10 Pc we
predict for flagged events). Unflagged events get -6.001, as the competition allowed.
Ideas follow the published top approaches (Uriot et al., Astrodynamics 2022): latest-risk based
predictions, demotion rules on object type / t_span / miss distance (team sesc), a constant value
for flagged events (team Magpies), and learnt corrections of the latest risk.
"""
from __future__ import annotations

import itertools
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

import kaksha as k
import train as T

OUT = pathlib.Path(__file__).resolve().parent / "results"
SMALL = dict(T.PARAMS, n_estimators=200)


# ---- decision rules: (Xa, ya, Xb) -> bool flags for Xb -------------------------------------------
def flag_latest(Xa, ya, Xb, promote=0.0, payload=False, span=None, miss=None):
    """Latest risk >= -6 - promote, then sesc-style demotions."""
    f = Xb.last_risk.to_numpy() >= k.HIGH - promote
    if payload:
        f &= Xb.object_type.astype(str).to_numpy() != "PAYLOAD"
    if span is not None:
        f &= ~(Xb.last_t_span.to_numpy() < span)
    if miss is not None:
        f &= ~(Xb.last_miss_distance.to_numpy() > miss)
    return f


def flag_combo(Xa, ya, Xb, promote=0.0, payload=False, span=None, miss=None, gbm_t=None):
    """sesc-style rules plus an optional classifier that promotes events just under the line."""
    f = flag_latest(Xa, ya, Xb, promote, payload, span, miss)
    if gbm_t is not None:
        lr = Xb.last_risk.to_numpy()
        band_a = (Xa.last_risk.to_numpy() >= k.HIGH - 1.0)
        clf = lgb.LGBMClassifier(**SMALL).fit(Xa[band_a], ya[band_a] >= k.HIGH)
        f |= (lr >= k.HIGH - 1.0) & (lr < k.HIGH) & (clf.predict_proba(Xb)[:, 1] >= gbm_t)
    return f


def flag_gbm(Xa, ya, Xb, t=0.5, promote=0.0):
    """Start from the latest-risk flags, demote those a classifier thinks end low (P(low) >= t)."""
    f = Xb.last_risk.to_numpy() >= k.HIGH - promote
    near = Xa.last_risk.to_numpy() >= k.HIGH - 1.0
    clf = lgb.LGBMClassifier(**SMALL).fit(Xa[near], ya[near] < k.HIGH)
    return f & (clf.predict_proba(Xb)[:, 1] < t)


# ---- value rules: (Xa, ya, Xb) -> predicted log10 Pc for Xb ---------------------------------------
def val_latest(Xa, ya, Xb):
    return Xb.last_risk.to_numpy()


def _hr_flagged(Xa, ya):
    return (Xa.last_risk.to_numpy() >= k.HIGH) & (ya >= k.HIGH)


def val_shift(Xa, ya, Xb):
    m = _hr_flagged(Xa, ya)
    return Xb.last_risk.to_numpy() + np.mean(ya[m] - Xa.last_risk.to_numpy()[m])


def val_linear(Xa, ya, Xb):
    m = _hr_flagged(Xa, ya)
    b, a = np.polyfit(Xa.last_risk.to_numpy()[m], ya[m], 1)
    return a + b * Xb.last_risk.to_numpy()


def val_const(Xa, ya, Xb):
    return np.full(len(Xb), ya[ya >= k.HIGH].mean())


def val_gbm_change(Xa, ya, Xb):
    near = Xa.last_risk.to_numpy() >= k.HIGH - 1.0
    hi = near & (ya >= k.HIGH)  # MSE only counts true high-risk events: learn their change
    r = lgb.LGBMRegressor(**dict(SMALL, min_child_samples=5)).fit(Xa[hi], ya[hi] - Xa.last_risk.to_numpy()[hi])
    return Xb.last_risk.to_numpy() + r.predict(Xb)


def predict(flag, value, Xa, ya, Xb):
    f = flag(Xa, ya, Xb)
    return np.where(f, np.maximum(value(Xa, ya, Xb), k.HIGH), k.CLIP)


def candidates():
    flags = {"latest": flag_latest}
    for p in (0.1, 0.2, 0.3):
        flags[f"latest promote {p}"] = lambda a, b, c, p=p: flag_latest(a, b, c, promote=p)
    flags["sesc demote payload"] = lambda a, b, c: flag_latest(a, b, c, payload=True)
    for s, m in itertools.product((None, 0.5, 1.0), (None, 30000, 15000)):
        if s is None and m is None:
            continue
        flags[f"sesc demote span<{s} miss>{m}"] = lambda a, b, c, s=s, m=m: flag_latest(a, b, c, span=s, miss=m)
    for p, pay, sp, mi, g in itertools.product((0.0, 0.1), (False, True), (None, 0.5), (None, 30000, 50000),
                                               (None, 0.3, 0.5)):
        name = f"combo promote={p} payload={pay} span<{sp} miss>{mi} gbm_promote={g}"
        flags[name] = lambda a, b, c, p=p, pay=pay, sp=sp, mi=mi, g=g: flag_combo(a, b, c, p, pay, sp, mi, g)
    for t in (0.5, 0.7, 0.9):
        flags[f"gbm demote P(low)>={t}"] = lambda a, b, c, t=t: flag_gbm(a, b, c, t=t)
    values = {"latest": val_latest, "latest+mean shift": val_shift, "linear in latest": val_linear,
              "constant (mean HR)": val_const, "gbm change": val_gbm_change}
    return flags, values


def main() -> None:
    X, y, *_ = T.prepare()  # only the train part is used below
    flags, values = candidates()
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=T.SEED).split(X, y >= k.HIGH))
    rows = []
    for (fn, f), (vn, v) in itertools.product(flags.items(), values.items()):
        if fn != "latest" and vn not in ("latest",) if fn.startswith("combo") else fn != "latest" and vn not in ("latest", "linear in latest"):
            continue  # value rules are compared on plain flags; flags on the two simplest values
        scores = []
        for rep in range(5):
            pred = np.zeros(len(y))
            for a, b in cv[rep * 5:(rep + 1) * 5]:
                pred[b] = predict(f, v, X.iloc[a], y[a], X.iloc[b])
            scores.append(k.score(y, pred))
        s = pd.DataFrame(scores)
        rows.append({"flag": fn, "value": vn, "L_mean": s.L.mean(), "L_std": s.L.std(), "mse_hr": s.mse_hr.mean(),
                     "f2": s.f2.mean(), "missed": s.missed_high_risk.mean(), "unnecessary": s.unnecessary_manoeuvres.mean()})
        print(f"{fn:34s} {vn:20s} L={s.L.mean():.3f}±{s.L.std():.3f} mse={s.mse_hr.mean():.3f} f2={s.f2.mean():.3f}")
    pd.DataFrame(rows).sort_values("L_mean").to_csv(OUT / "validation.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
