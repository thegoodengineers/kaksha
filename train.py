"""Train, evaluate and write every reported number to results/*.csv.

Two protocols, both split by event (one event's CDMs never sit on both sides):
  official  ESA's own split: fit on ESA's train events, score on ESA's 2,167 held-out test events.
  shifted   ESA's test events in 5 stratified folds; each fold is scored by a model fitted on ESA's
            train events plus the other 4 folds. Every test event is predicted by a model that never
            saw it. Exists because ESA's train and test events behave differently (see
            results/data_summary.csv), and an operator would retrain on recent events.
In both, the decision threshold and the probability calibration are chosen from out-of-fold
predictions on the training side only, never from the events being scored.

Model: gradient-boosted trees (LightGBM; Ke et al., NeurIPS 2017) for P(final log10 Pc >= -6),
isotonic calibration (Zadrozny & Elkan, KDD 2002), a binned empirical 10-90 % interval.
Imbalance is handled by choosing the threshold that maximises F2 (recall-weighted) rather than
by resampling.
"""
from __future__ import annotations

import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

import kaksha as k

OUT = pathlib.Path(__file__).resolve().parent / "results"
SEED = 0
FLOOR = -10.0  # risk shown as "below -10" in intervals; such events almost never end high
P_BINS = np.array([0, 0.05, 0.2, 0.5, 1.0001])  # probability groups for the interval
GRID = np.round(np.arange(0.01, 0.96, 0.01), 2)
PARAMS = dict(n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
              random_state=SEED, verbose=-1)
LABELS = {  # plain words for the reasons shown to an operator
    "last_risk": "latest risk", "risk_max": "highest risk so far", "risk_change": "change in risk",
    "risk_std": "risk spread", "risk_first": "first risk", "n_cdms": "number of messages",
    "last_miss_distance": "miss distance (m)", "miss_distance_change": "change in miss distance",
    "last_max_risk_estimate": "worst-case risk estimate", "last_time_to_tca": "days to closest approach",
    "last_mahalanobis_distance": "Mahalanobis distance", "mission_id": "mission",
    "object_type": "debris object type", "last_relative_speed": "relative speed (m/s)",
}


def label(name: str) -> str:
    return LABELS.get(name, name.replace("last_", "latest ").replace("_", " "))


def prepare():
    cdms, y = k.split_cdms(k.load("train_data"))
    X = k.features(cdms)
    test = k.load("test_data")
    Xt = k.features(test)[X.columns]
    yt = k.load("test_data_private").set_index("event_id").true_risk.reindex(Xt.index)
    for c in ("mission_id", "object_type"):
        cats = sorted(set(X[c].astype(str)) | set(Xt[c].astype(str)))
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
        Xt[c] = pd.Categorical(Xt[c].astype(str), categories=cats)
    return X, y.reindex(X.index).to_numpy(), Xt, yt.to_numpy(), test


def folds(y, n=5):
    return StratifiedKFold(n, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y >= k.HIGH)


def classifier(X, y):
    return lgb.LGBMClassifier(**PARAMS).fit(X, y >= k.HIGH)


def best_threshold(p, high) -> float:
    return float(GRID[np.argmax([k.fbeta(high, p >= t) for t in GRID])])


def fit(X, y, target_like):
    """Fit on (X, y). Calibration and threshold come from 5-fold out-of-fold predictions on the
    rows flagged target_like (the events that look like the ones we will score)."""
    oof = np.zeros(len(y))
    for a, b in folds(y):
        oof[b] = classifier(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[target_like], y[target_like] >= k.HIGH)
    thr = best_threshold(iso.predict(oof[target_like]), y[target_like] >= k.HIGH)
    # Interval: empirical 10/50/90 % quantiles of (final - latest risk) among out-of-fold events
    # like the ones we score, grouped by calibrated probability (a binned split-conformal
    # interval; Lei et al., JASA 2018). Events already below FLOOR get the floor.
    latest = X.last_risk.to_numpy()
    rows = target_like & (latest >= FLOOR)
    change = (np.maximum(y, FLOOR) - latest)[rows]
    group = np.digitize(iso.predict(oof[rows]), P_BINS) - 1
    qs = np.array([np.quantile(change[group == g], [0.1, 0.5, 0.9]) for g in range(len(P_BINS) - 1)])
    return {"clf": classifier(X, y), "iso": iso, "thr": thr, "q": qs}


def predict(m, X):
    p = m["iso"].predict(m["clf"].predict_proba(X)[:, 1])
    latest = X.last_risk.to_numpy()
    cand = latest >= FLOOR
    qc = m["q"][np.digitize(p, P_BINS) - 1]
    q = {a: np.where(cand, np.maximum(latest + qc[:, i], FLOOR), FLOOR) for i, a in enumerate((0.1, 0.5, 0.9))}
    contrib = m["clf"].booster_.predict(X, pred_contrib=True)[:, :-1]
    names = X.columns
    reasons = []
    for i in range(len(X)):
        top = np.argsort(-np.abs(contrib[i]))[:3]
        reasons.append("; ".join(
            f"{label(names[j])} {fmt(X.iloc[i, j])} ({'raises' if contrib[i, j] > 0 else 'lowers'} risk)"
            for j in top))
    manoeuvre = p >= m["thr"]
    # Forecast value: the latest risk, lifted to the line when we recommend a manoeuvre and
    # clipped below it when we don't (the competition's -6.001 rule).
    risk = np.where(manoeuvre, np.maximum(latest, k.HIGH), k.CLIP)
    return pd.DataFrame({"p_high": p, "threshold": m["thr"], "manoeuvre": manoeuvre,
                         "pred_risk": risk, "q10": q[0.1], "q50": q[0.5], "q90": q[0.9],
                         "reasons": reasons}, index=X.index)


def fmt(v) -> str:
    if isinstance(v, str) or not np.isfinite(v):
        return str(v)
    return f"{v:.3g}"


def metrics_row(protocol, method, y, risk, manoeuvre=None):
    s = k.score(y, risk)
    if manoeuvre is not None:  # decision counts follow the recommendation, not the clipped value
        s.update(k.decision_counts(y >= k.HIGH, manoeuvre))
    return {"protocol": protocol, "method": method, **s}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    X, y, Xt, yt, test = prepare()
    latest_t = Xt.last_risk.to_numpy()
    rows = [metrics_row(p, "baseline: latest CDM risk", yt, k.clip(latest_t)) for p in ("official", "shifted")]
    rows += [metrics_row(p, "baseline: highest risk so far", yt, k.clip(Xt.risk_max)) for p in ("official", "shifted")]

    off = predict(fit(X, y, np.ones(len(y), bool)), Xt)
    rows.append(metrics_row("official", "LightGBM gate", yt, off.pred_risk, off.manoeuvre))
    # The two candidates frozen from train-only validation (tune.py) and logged in results/test_log.csv.
    # Recomputed here for metrics.csv: same predictions, not new test evaluations.
    from evaluate_official import CANDIDATES
    import tune
    for name, cfg in CANDIDATES.items():
        f = tune.flag_combo(X, y, Xt, **cfg)
        rows.append(metrics_row("official", f"rules: {name}", yt, np.where(f, np.maximum(latest_t, k.HIGH), k.CLIP)))

    parts = []
    for fold, (a, b) in enumerate(folds(yt)):
        Xa = pd.concat([X, Xt.iloc[a]])
        ya = np.r_[y, yt[a]]
        like = np.r_[np.zeros(len(y), bool), np.ones(len(a), bool)]
        part = predict(fit(Xa, ya, like), Xt.iloc[b])
        part["fold"] = fold
        parts.append(part)
        print("fold", fold, "threshold", part.threshold.iloc[0])
    sh = pd.concat(parts).reindex(Xt.index)
    rows.append(metrics_row("shifted", "LightGBM gate", yt, sh.pred_risk, sh.manoeuvre))
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False, float_format="%.4f")

    high = yt >= k.HIGH
    # Calibration of the shifted protocol's probabilities.
    bins = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001])
    idx = np.digitize(sh.p_high, bins) - 1
    pd.DataFrame([{"bin_low": bins[i], "bin_high": min(bins[i + 1], 1.0), "events": int((idx == i).sum()),
                   "mean_predicted": float(sh.p_high[idx == i].mean()) if (idx == i).any() else np.nan,
                   "observed_high_risk": float(high[idx == i].mean()) if (idx == i).any() else np.nan}
                  for i in range(len(bins) - 1)]).to_csv(OUT / "calibration.csv", index=False, float_format="%.4f")
    # Trade-off curve: what each threshold would cost on these events (for the console slider;
    # the reported operating point is the threshold chosen inside each fold, above).
    trade = [{"threshold": t, **k.decision_counts(high, sh.p_high.to_numpy() >= t)} for t in GRID]
    pd.DataFrame(trade).to_csv(OUT / "tradeoff.csv", index=False, float_format="%.4f")
    # Uncertainty: how often the final risk (floored at -10) falls inside the 10-90 % interval.
    yf = np.maximum(yt, FLOOR)
    inside = (yf >= sh.q10) & (yf <= sh.q90)
    pd.DataFrame([{"protocol": "shifted", "nominal": 0.8, "coverage_all": inside.mean(),
                   "coverage_high_risk": inside[high].mean(),
                   "brier": float(np.mean((sh.p_high.to_numpy() - high) ** 2)),
                   "brier_latest_rule": float(np.mean((latest_t >= k.HIGH) != high))}]
                 ).to_csv(OUT / "uncertainty.csv", index=False, float_format="%.4f")
    # Data summary: why there are two protocols.
    summ = []
    for name, XX, yy in (("ESA train (usable events)", X, y), ("ESA test", Xt, yt)):
        lr = XX.last_risk.to_numpy()
        summ.append({"set": name, "events": len(yy), "cdms_used": int(XX.n_cdms.sum()),
                     "high_risk_final": int((yy >= k.HIGH).sum()),
                     "latest_high": int((lr >= k.HIGH).sum()),
                     "latest_high_final_low": int(((lr >= k.HIGH) & (yy < k.HIGH)).sum()),
                     "latest_low_final_high": int(((lr < k.HIGH) & (yy >= k.HIGH)).sum())})
    pd.DataFrame(summ).to_csv(OUT / "data_summary.csv", index=False)
    sh.assign(true_final_risk=yt, latest_risk=latest_t, n_cdms=Xt.n_cdms.to_numpy(),
              mission_id=Xt.mission_id.astype(str).to_numpy()).to_csv(
        OUT / "predictions.csv", index_label="event_id", float_format="%.4f")
    print(pd.DataFrame(rows)[["protocol", "method", "L", "mse_hr", "f2", "missed_high_risk",
                              "unnecessary_manoeuvres"]].to_string(index=False))


if __name__ == "__main__":
    main()
