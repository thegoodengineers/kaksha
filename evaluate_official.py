"""Score ONE frozen candidate on ESA's official test set and append the result to results/test_log.csv.

Every run is a test peek and is logged, so the number of peeks is visible. Candidates were chosen
on train-only validation (tune.py, results/validation.csv) before this script was run.
Usage: python evaluate_official.py <candidate>  |  python evaluate_official.py --bootstrap <candidate>
"""
import datetime
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

import kaksha as k
import train as T
import tune

LOG = pathlib.Path(__file__).resolve().parent / "results" / "test_log.csv"
CANDIDATES = {
    # Headline. Frozen from results/cv_selection.csv: lowest mean L over one event-grouped 5-fold CV on
    # ESA's train events only (tune.py --select); ties broken toward the simpler config (no classifier).
    "clean": dict(promote=0.1, payload=True, span=1.0, miss=15000, gbm_t=None),
    # Earlier, frozen from results/validation.csv after ~60 unlogged exploratory test scorings.
    "rules": dict(promote=0.1, payload=True, span=0.5, miss=30000, gbm_t=None),
    "rules+gbm_promote": dict(promote=0.0, payload=True, span=0.5, miss=30000, gbm_t=0.3),
}


def flags(name, X, y, Xt):
    return tune.flag_combo(X, y, Xt, **CANDIDATES[name])


def main(name: str) -> None:
    X, y, Xt, yt, _ = T.prepare()
    f = flags(name, X, y, Xt)
    s = k.score(yt, np.where(f, np.maximum(Xt.last_risk.to_numpy(), k.HIGH), k.CLIP))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    label = "clean: chosen by train CV only" if name == "clean" else name
    row = {"date": datetime.date.today().isoformat(), "commit": commit, "candidate": label,
           "config": str(CANDIDATES[name]), **{c: s[c] for c in ("L", "mse_hr", "f2", "missed_high_risk",
                                                                  "unnecessary_manoeuvres")}}
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False, float_format="%.4f")
    print(row)




def bootstrap(name: str, n: int = 2000) -> None:
    """Paired bootstrap over test events of L(candidate) - L(latest rule), for an already-logged
    candidate. Re-uses the same frozen predictions; nothing is fitted or chosen here."""
    X, y, Xt, yt, _ = T.prepare()
    ours = np.where(flags(name, X, y, Xt), np.maximum(Xt.last_risk.to_numpy(), k.HIGH), k.CLIP)
    base = k.clip(Xt.last_risk.to_numpy())
    rng = np.random.default_rng(T.SEED)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        d.append(k.score(yt[i], ours[i])["L"] - k.score(yt[i], base[i])["L"])
    d = np.array(d)
    out = pd.DataFrame([{"candidate": name, "resamples": n, "delta_L_mean": d.mean(),
                         "delta_L_p2.5": np.percentile(d, 2.5), "delta_L_p97.5": np.percentile(d, 97.5),
                         "share_better": (d < 0).mean()}])
    out.to_csv(LOG.parent / "official_bootstrap.csv", index=False, float_format="%.4f")
    print(out.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "--bootstrap":
        bootstrap(sys.argv[2])
    else:
        main(sys.argv[1])
