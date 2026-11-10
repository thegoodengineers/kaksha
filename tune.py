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




def cv_select() -> dict:
    """Clean selection: one event-grouped stratified 5-fold CV on ESA's train events over the full rule
    search space; the winner is the config with the lowest mean fold L. Writes every config and its
    fold scores to results/cv_selection.csv. The test set is not read."""
    from sklearn.model_selection import StratifiedKFold
    X, y, *_ = T.prepare()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=T.SEED).split(X, y >= k.HIGH))
    rows = []
    for p, pay, sp, mi, g in itertools.product((0.0, 0.1, 0.2, 0.3), (False, True), (None, 0.5, 1.0),
                                               (None, 15000, 30000, 50000), (None, 0.3, 0.5)):
        cfg = dict(promote=p, payload=pay, span=sp, miss=mi, gbm_t=g)
        Ls = []
        for a, b in folds:
            f = flag_combo(X.iloc[a], y[a], X.iloc[b], **cfg)
            Ls.append(k.score(y[b], np.where(f, np.maximum(X.last_risk.to_numpy()[b], k.HIGH), k.CLIP))["L"])
        rows.append({**{key: ("" if v is None else v) for key, v in cfg.items()},
                     **{f"L_fold{i}": v for i, v in enumerate(Ls)}, "L_mean": float(np.mean(Ls))})
    d = pd.DataFrame(rows).sort_values("L_mean", kind="stable")
    d.to_csv(OUT / "cv_selection.csv", index=False, float_format="%.4f")
    best = d.iloc[0]
    print(d.head(5).to_string(index=False))
    return {key: (None if best[key] == "" else best[key]) for key in ("promote", "payload", "span", "miss", "gbm_t")}



# ---- Shrunk-correction family (second clean search) ---------------------------------------------
# final = latest + lam * ridge correction, so lam = 0 is exactly the latest-CDM baseline and the model
# can only move away from it when cross-validation says so. Optional classifiers for the two failure
# cases: demote latest-high events likely to end low, promote latest-low events (within 1) likely to
# end high. Ridge regression: Hoerl & Kennard, Technometrics 1970.
RIDGE_COLS = ["last_risk", "risk_change", "risk_max", "risk_min", "risk_std", "risk_first", "n_cdms",
              "last_time_to_tca", "last_max_risk_estimate", "last_max_risk_scaling", "last_mahalanobis_distance",
              "mahalanobis_distance_change", "last_miss_distance", "miss_distance_change", "last_t_span",
              "last_c_sigma_t", "last_c_sigma_n", "c_sigma_t_change", "c_sigma_n_change", "last_relative_speed"]


def _design(X):
    from sklearn.impute import SimpleImputer  # noqa: F401  (kept local: only this family needs it)
    D = X[RIDGE_COLS].astype(float).copy()
    for c in ("last_miss_distance", "last_c_sigma_t", "last_c_sigma_n", "last_t_span"):
        D[c] = np.log10(D[c].clip(lower=1e-3))
    D = D.replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)
    for t in ("DEBRIS", "PAYLOAD", "UNKNOWN", "ROCKET BODY"):
        D[f"obj_{t}"] = (X.object_type.astype(str) == t).astype(float)
    return D


SUBSETS = {"latest>=-8": lambda X: X.last_risk.to_numpy() >= -8,
           "latest>=-6.5": lambda X: X.last_risk.to_numpy() >= -6.5,
           "latest>=-8 & last CDM<=2.5 d": lambda X: (X.last_risk.to_numpy() >= -8) & (X.last_time_to_tca.to_numpy() <= 2.5)}


def correction(Xa, ya, Xb, subset, alpha):
    """Ridge on (clipped final - clipped latest) over the training subset."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = SUBSETS[subset](Xa)
    target = np.maximum(ya, k.CLIP) - np.maximum(Xa.last_risk.to_numpy(), k.CLIP)
    model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=alpha)).fit(_design(Xa)[m], target[m])
    return model.predict(_design(Xb))


def failure_probs(Xa, ya, Xb):
    """P(final low) for latest-high events and P(final high) for latest in [-7, -6)."""
    la, lb = Xa.last_risk.to_numpy(), Xb.last_risk.to_numpy()
    hi = la >= k.HIGH
    p_low = lgb.LGBMClassifier(**SMALL).fit(Xa[hi], ya[hi] < k.HIGH).predict_proba(Xb)[:, 1]
    near = (la >= k.HIGH - 1) & (la < k.HIGH)
    p_up = lgb.LGBMClassifier(**dict(SMALL, min_child_samples=5)).fit(Xa[near], ya[near] >= k.HIGH).predict_proba(Xb)[:, 1]
    return np.where(lb >= k.HIGH, p_low, 0.0), np.where((lb >= k.HIGH - 1) & (lb < k.HIGH), p_up, 0.0)


def combine(latest, corr, p_low, p_up, lam, demote_t, promote_t):
    pred = latest + lam * corr
    if demote_t is not None:
        pred = np.where(p_low >= demote_t, np.minimum(pred, k.CLIP), pred)
    if promote_t is not None:
        pred = np.where(p_up >= promote_t, np.maximum(pred, k.HIGH), pred)
    return k.clip(pred)


CORR_GRID = dict(subset=list(SUBSETS), alpha=[1.0, 10.0, 100.0, 1000.0], lam=[0.0, 0.25, 0.5, 1.0],
                 demote_t=[None, 0.8, 0.9], promote_t=[None, 0.5, 0.7])


def fit_predict_correction(Xa, ya, Xb, cfg):
    p_low, p_up = failure_probs(Xa, ya, Xb)
    corr = correction(Xa, ya, Xb, cfg["subset"], cfg["alpha"])
    return combine(Xb.last_risk.to_numpy(), corr, p_low, p_up, cfg["lam"], cfg["demote_t"], cfg["promote_t"])


def cv_select_correction() -> dict:
    """One event-grouped stratified 5-fold CV on ESA's train events; lowest mean fold L wins; ties go to
    the config listed first (smaller lam, no classifiers). Writes results/cv_selection_correction.csv."""
    from sklearn.model_selection import StratifiedKFold
    X, y, *_ = T.prepare()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=T.SEED).split(X, y >= k.HIGH))
    cache = []
    for a, b in folds:  # everything that depends only on the fold is fitted once
        p_low, p_up = failure_probs(X.iloc[a], y[a], X.iloc[b])
        corr = {(s, al): correction(X.iloc[a], y[a], X.iloc[b], s, al)
                for s in CORR_GRID["subset"] for al in CORR_GRID["alpha"]}
        cache.append((b, p_low, p_up, corr))
    rows = []
    for vals in itertools.product(*CORR_GRID.values()):
        cfg = dict(zip(CORR_GRID, vals))
        Ls = [k.score(y[b], combine(X.last_risk.to_numpy()[b], corr[(cfg["subset"], cfg["alpha"])], p_low, p_up,
                                    cfg["lam"], cfg["demote_t"], cfg["promote_t"]))["L"]
              for b, p_low, p_up, corr in cache]
        rows.append({**{key: ("" if v is None else v) for key, v in cfg.items()},
                     **{f"L_fold{i}": v for i, v in enumerate(Ls)}, "L_mean": float(np.mean(Ls))})
    d = pd.DataFrame(rows).sort_values("L_mean", kind="stable")
    d.to_csv(OUT / "cv_selection_correction.csv", index=False, float_format="%.4f")
    print(d.head(8).to_string(index=False))
    base = d[(d.lam == 0) & (d.demote_t == "") & (d.promote_t == "")].iloc[0]
    print("baseline-equivalent config L_mean:", round(base.L_mean, 4))
    best = d.iloc[0]
    return {key: (None if best[key] == "" else best[key]) for key in CORR_GRID}


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--select-correction"]:
        print("CV pick:", cv_select_correction())
    elif sys.argv[1:] == ["--select"]:
        print("CV pick:", cv_select())
    else:
        main()
