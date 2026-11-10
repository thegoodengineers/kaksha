# Chirag (team leader): eligibility, model, final deck, Unstop upload

## Rule for everyone: never name the tools we used to build this
Do not name any assistant, chatbot or code tool used to build this project anywhere: commits, co-author lines,
code comments, README, docs, the deck, screenshots or videos. No `Co-Authored-By` lines for tools.
Before you push: `git log --format=%B | grep -i co-authored` prints nothing.
Branch: `<yourname>/<task>` → pull request into `main`. Run `python -m pytest -q` before every push.

## Done
- [x] Repo, data download script, 2-day cutoff, features, both evaluation protocols, results CSVs.
- [x] Console, tests, CI. Deck draft in `deck/StarForge_SW5_TheGoodEngineers.pptx` and `.pdf`.

## 1. Eligibility (by 18 Nov) - needs a person
- [ ] Confirm the team on Unstop is 3-4 of Chirag, Bhumika, Ayush, Abhijeet **and includes at least one female
      member (Bhumika)**, or the team is not eligible. If all four are registered, that is fine (max 4).
- [ ] Ask the organisers (Unstop discussion or the coordinator) whether pre-event analysis on public data is
      allowed, and save their answer in the team chat. Our deck says the prototype exists.

## 2. Upload (by 20 Nov, 11:59 PM IST; aim for 19 Nov night) - only the team leader can
- [ ] Open `deck/StarForge_SW5_TheGoodEngineers.pdf`, check every slide once more.
- [ ] Upload it on Unstop (PPT, PPTX or PDF accepted). Screenshot the confirmation into the team chat.

## 3. Model (if time before 12 Dec)
- [ ] Ranges for high-risk events: 70 % coverage for an 80 % range. Try per-group widths or conformal by
      latest-risk band. Re-run `train.py`, update README and deck from the CSVs.
