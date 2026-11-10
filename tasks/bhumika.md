# Bhumika: console design and accessibility

## Rule for everyone: never name the tools we used to build this
Do not name any assistant, chatbot or code tool used to build this project anywhere: commits, co-author lines,
code comments, README, docs, the deck, screenshots or videos. No `Co-Authored-By` lines for tools.
Before you push: `git log --format=%B | grep -i co-authored` prints nothing.
Branch: `<yourname>/<task>` → pull request into `main`. Run `python -m pytest -q` before every push.

## Done
- [x] Console at `web/index.html`: event list, risk history, recommendation, reasons, reveal, trade-off slider,
      calibration table. 0 axe-core problems at 1440 px and 400 px.

## 1. Confirm eligibility with Chirag (by 18 Nov) - needs a person
- [ ] Check you are on the Unstop team page. The team needs at least one female member to be eligible.

## 2. Look at it like an operator (by 19 Nov)
- [ ] Run the console (README "Run it"), open events 1573, 873 and a "Missed by KAKSHA" one. Note anything
      confusing in the team chat. The interval line ("80 % range") is the most likely thing to confuse.
- [ ] mobbin.com blocked automated browsing: look at 2-3 real monitoring dashboards by hand and add what you
      take to `design/REFS.md`.

## 3. Round 2 (12-13 Dec)
- [ ] Owner of the console polish block (H10-H16 in the hour plan on the deck's build-plan slide).
