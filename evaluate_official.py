"""Score ONE frozen candidate on ESA's official test set and append the result to results/test_log.csv.

Every run is a test peek and is logged, so the number of peeks is visible. Candidates were chosen
on train-only validation (tune.py, results/validation.csv) before this script was run.
Usage: python evaluate_official.py <candidate>
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
CANDIDATES = {  # frozen from results/validation.csv (rank 1 and the best rules-only row)
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
    row = {"date": datetime.date.today().isoformat(), "commit": commit, "candidate": name,
           "config": str(CANDIDATES[name]), **{c: s[c] for c in ("L", "mse_hr", "f2", "missed_high_risk",
                                                                  "unnecessary_manoeuvres")}}
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False, float_format="%.4f")
    print(row)


if __name__ == "__main__":
    main(sys.argv[1])
