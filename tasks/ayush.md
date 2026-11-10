# Ayush: deck and pitch

## Rule for everyone: never name the tools we used to build this
Do not name any assistant, chatbot or code tool used to build this project anywhere: commits, co-author lines,
code comments, README, docs, the deck, screenshots or videos. No `Co-Authored-By` lines for tools.
Before you push: `git log --format=%B | grep -i co-authored` prints nothing.
Branch: `<yourname>/<task>` → pull request into `main`. Run `python -m pytest -q` before every push.

## Done
- [x] Deck filled in the official template with real screenshots and numbers from `results/*.csv`:
      `deck/StarForge_SW5_TheGoodEngineers.pptx` and `.pdf` (7 template slides + 2 extra).

## 1. Review the deck (by 19 Nov) - needs a person
- [ ] Read every slide out loud in under 3 minutes. Cut words, not numbers. Every number must match
      `docs/RESULTS.md`.
- [ ] Check the team name, leader, track and code on slide 1 against Unstop.
- [ ] If you change the PPTX, export the PDF again from PowerPoint and check it page by page.

## 2. Pitch (before 12 Dec)
- [ ] 20-second plain-English problem opening, then the event 1573 story (latest message just under the line,
      KAKSHA says manoeuvre, the final risk was 10x above the line).
- [ ] Say the honest part: on ESA's original split alone the model loses to the simple rule; explain why.
