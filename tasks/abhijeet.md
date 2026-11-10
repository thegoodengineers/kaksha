# Abhijeet: data checks, tests, CI

## Rule for everyone: never name the tools we used to build this
Do not name any assistant, chatbot or code tool used to build this project anywhere: commits, co-author lines,
code comments, README, docs, the deck, screenshots or videos. No `Co-Authored-By` lines for tools.
Before you push: `git log --format=%B | grep -i co-authored` prints nothing.
Branch: `<yourname>/<task>` → pull request into `main`. Run `python -m pytest -q` before every push.

## Done
- [x] `scripts/get_data.py` (MD5-checked download), `test_kaksha.py` (cutoff, split, metric, published baseline
      reproduced), GitHub Actions running all of it with the dataset cached.

## 1. Second pair of eyes on leakage (by 19 Nov) - needs a person
- [ ] Read `kaksha.py` and `train.py` and try to find any way a message under 2 days, or the same event,
      reaches training for the events being scored. Write what you checked in a pull request.
- [ ] Re-run `python train.py` on your laptop and compare `results/metrics.csv` with the repo. Note any
      difference (LightGBM can differ slightly across machines).

- [ ] Check `results/test_log.csv`: every official-test evaluation must be there. The headline is the `clean` row. Any new test run goes through
      `evaluate_official.py` so it is logged; choose candidates with `tune.py` (train events only) first.

## 2. Round 2 (12-13 Dec)
- [ ] Owner of data pipeline and evaluation (H0-H3 and H16-H20 in the hour plan).
