# Working on KAKSHA

StarForge 2026 (BMSITM, Bengaluru), Software track, problem SW-5. Team: The Good Engineers.

## Dates (IST)
| By | What |
|---|---|
| 18 Nov, 9:00 PM | Everyone has run the demo once and read the deck; comments in the team chat |
| 19 Nov, 9:00 PM | Deck final (`deck/`); team details checked on Unstop |
| 20 Nov, 11:59 PM | **Round 1 deadline.** Chirag uploads the deck PDF on Unstop. We upload on 19 Nov night |
| before 12 Dec | Round 1 results |
| 12 Dec 10 AM - 13 Dec 10 AM | Round 2: 24-hour build in person at BMSITM, then demos |

## Setup (once)
```
git clone https://github.com/thegoodengineers/kaksha.git
cd kaksha
python -m pip install -r requirements.txt
python scripts/get_data.py                 # ESA dataset into data/ (gitignored, never commit it)
python -m pytest -q                        # must pass before you start
python train.py                            # about 2 minutes
python -m uvicorn server:app --port 8780   # console at http://localhost:8780
```

## Rules
- **Never name any tool, assistant or chatbot used to build this project**: not in commits, co-author lines,
  comments, docs, the deck, screenshots or videos. Before you push,
  `git log --format=%B | grep -i co-authored` must print nothing.
- Branch per task: `yourname/short-task`, pull request into `main`. Run `python -m pytest -q` first.
- **No invented numbers.** Every number in the README, deck or pitch comes from `results/*.csv` (written
  by `train.py`) or a cited paper, labelled as such. Re-run `train.py` and `scripts/results_md.py` after
  any model change and update the deck from the CSVs.
- **No leakage.** Nothing may read a message closer than 2 days to closest approach, and no event may be
  on both sides of a split. `test_kaksha.py` checks both; add a test with any change to `kaksha.py`.
- Say where we lose. The official-split result (model worse than the rule) stays in the README and deck.
- Data credit: ESA, CC-BY-4.0. Never commit `data/`.

## Who owns what
| Part | Files | Owner |
|---|---|---|
| Model, evaluation, submission | `train.py`, `kaksha.py`, `results/` | Chirag |
| Console and design | `web/index.html`, `design/` | Bhumika |
| Deck and pitch | `deck/`, `docs/screenshots/` | Ayush |
| Data checks, tests, CI | `scripts/`, `test_kaksha.py`, `.github/` | Abhijeet |

## Your tasks: one file each
[`tasks/chirag.md`](tasks/chirag.md) · [`tasks/bhumika.md`](tasks/bhumika.md) ·
[`tasks/ayush.md`](tasks/ayush.md) · [`tasks/abhijeet.md`](tasks/abhijeet.md)
