# KAKSHA

Two days before closest approach: plan a manoeuvre, or keep monitoring?

Satellite operators get conjunction data messages (CDMs) warning of close approaches, with a
collision probability (Pc) that is updated for days. They must decide early whether to spend fuel on an
avoidance manoeuvre. KAKSHA reads only the messages received at least two days before closest
approach, forecasts the final risk, and recommends **plan a manoeuvre** or **monitor**, with a calibrated
probability, a range for the final risk, and the three things that drove the call.

StarForge 2026 (BMSITM), Software track, problem SW-5. Team: The Good Engineers.

![KAKSHA console](docs/screenshots/console-top.png)

## Results on ESA's held-out events
Data: ESA's Collision Avoidance Challenge dataset, real anonymised CDMs from 2015-2019
(Zenodo 10.5281/zenodo.4463683, CC-BY-4.0). 2,167 held-out events, 150 of which end high risk
(log10 Pc >= -6). Scored with the competition's own metric (Uriot et al., Astrodynamics 2022). Every number
below is in `results/*.csv`, written by `train.py`; the full tables are in [`docs/RESULTS.md`](docs/RESULTS.md).

| On 2,167 held-out events | Latest-message rule | KAKSHA |
|---|---|---|
| High-risk events missed (of 150) | 35 | **23** |
| Unnecessary manoeuvres | 63 | 100 |
| ESA score L (lower is better) | 0.694 | **0.674** |
| F2 (recall-weighted) | 0.739 | **0.768** |

KAKSHA misses 12 fewer dangerous events and pays for it with 37 more manoeuvres on events that end safe.
The slider in the console moves along that trade-off.

**Read this before quoting the table.** These numbers are from the *shifted* protocol: the test events are
split into 5 folds by event, and each fold is predicted by a model trained on ESA's train events plus the
other 4 folds. No event is ever scored by a model that saw any of its messages, and the decision line and
calibration are chosen from training folds only. We need it because ESA's train and test events behave
differently: of the events whose latest message said "high risk", 85 % ended low in the train set (261 of
308) but 35 % in the test set (63 of 178) (`results/data_summary.csv`). On ESA's original split alone (train
on ESA's train events, score the test events) **KAKSHA is worse than the rule**: L 0.849, 38 missed. We say
so in the console too.

Our metric code reproduces the paper's published score for the latest-message rule exactly (L = 0.694 on
2,167 events, 150 high risk; checked in `test_kaksha.py`). The competition winner scored 0.556 (published);
we have not matched that.

Also measured: the probabilities are calibrated (events given 0.6-0.8 ended high risk 70 % of the time;
Brier 0.032 against 0.045 for the rule) and the 80 % range held the final risk for 94 % of all events but only
70 % of high-risk ones (`results/calibration.csv`, `results/uncertainty.csv`).

## How it works
1. `scripts/get_data.py` downloads the dataset (MD5-checked) into `data/` (never committed).
2. `kaksha.py` keeps events whose last message is within 1 day of closest approach, drops every message
   closer than 2 days, and builds one row per event: the latest message plus trends over the history
   (154 features). It refuses any message inside the cutoff.
3. `train.py` fits gradient-boosted trees for P(final risk high), calibrates them with isotonic regression,
   picks the manoeuvre line that maximises F2 on out-of-fold training predictions (recall weighs 4x
   precision, as in the competition), and builds an empirical 10-90 % range for the final risk.
   Reasons come from per-event feature contributions of the tree model.
4. `server.py` + `web/index.html`: the operator console. Pick an event, see its risk history with the
   2-day cutoff shaded, the recommendation and reasons, then "Reveal what happened".

## Run it
```
python -m pip install -r requirements.txt
python scripts/get_data.py          # 221 MB download from Zenodo
python -m pytest -q                 # split, cutoff and metric checks
python train.py                     # about 2 minutes; writes results/*.csv
python scripts/results_md.py        # docs/RESULTS.md from the CSVs
python -m uvicorn server:app --port 8780   # http://localhost:8780
```
Our page scores 0 axe-core accessibility problems (checked with our `a11y.py`, desktop and mobile widths).

## Known limits (honest status)
- Working today: data pipeline, both evaluation protocols, calibrated recommendation, console.
- Ranges are too narrow for high-risk events (70 % coverage for an 80 % range).
- Most catches are events sitting near the line; events that jump from far below it are still missed.
- The demo event (1573) was picked after the fact to show a catch; the console's filters show misses and
  unnecessary manoeuvres just as easily.
- A research prototype on anonymised historical data, not an operational system.

## Next (24-hour round, if shortlisted)
Sequence models over the message history, a cost slider in fuel and Pc terms, per-mission calibration,
better ranges for high-risk events, and a live CDM feed format (CCSDS CDM).

## Team
How we work and who owns what: [`CONTRIBUTING.md`](CONTRIBUTING.md). Each person's tasks: [`tasks/`](tasks/).
Design references: [`design/REFS.md`](design/REFS.md). Deck: [`deck/`](deck/).
