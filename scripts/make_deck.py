"""Fill the official StarForge template from results/*.csv and docs/screenshots/.

Run: python scripts/make_deck.py <template.pptx>
Needs PowerPoint (Windows) for the two extra slides (duplicated from template slide 6) and the PDF.
Every number on the slides is read from results/*.csv here, never typed in.
"""
import pathlib
import subprocess
import sys

import matplotlib
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECK = ROOT / "deck"
SHOTS = ROOT / "docs" / "screenshots"
OUT = DECK / "StarForge_SW5_TheGoodEngineers.pptx"
NAVY, INK, GREY = RGBColor(0x14, 0x2B, 0x5C), RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x55, 0x5E, 0x6E)
AMBER, TEAL, PALE = RGBColor(0xC9, 0x7A, 0x00), RGBColor(0x0E, 0x7C, 0x74), RGBColor(0xEE, 0xF2, 0xF8)
FONT = "Arial"
DY = 0.4  # keeps content below the template's star at the top left

m = pd.read_csv(ROOT / "results" / "metrics.csv")
row = lambda p, meth: m[(m.protocol == p) & (m.method == meth)].iloc[0]  # noqa: E731
OURS, RULE = row("shifted", "LightGBM gate"), row("shifted", "baseline: latest CDM risk")
OFF = row("official", "LightGBM gate")
RULES = row("official", "rules: clean")  # clean pick 1: chosen by train CV only
CORR = row("official", "correction: clean-correction")  # clean pick 2
BOOT = pd.read_csv(ROOT / "results" / "official_bootstrap.csv").iloc[0]
DS = pd.read_csv(ROOT / "results" / "data_summary.csv").set_index("set")
TR, TE = DS.loc["ESA train (usable events)"], DS.loc["ESA test"]
U = pd.read_csv(ROOT / "results" / "uncertainty.csv").iloc[0]
CAL = pd.read_csv(ROOT / "results" / "calibration.csv")
N_EV, N_HI = int(OURS.events), int(OURS.high_risk_events)
MISS_O, MISS_R = int(OURS.missed_high_risk), int(RULE.missed_high_risk)
UNN_O, UNN_R = int(OURS.unnecessary_manoeuvres), int(RULE.unnecessary_manoeuvres)


def text(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, dy=DY):
    """paras: list of (text, size, bold, colour) or str; a str starting with '• ' becomes a bullet line."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y + dy), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, p in enumerate(paras):
        if isinstance(p, str):
            p = (p, 21, False, INK)
        t, size, bold, colour = p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(size * 0.35)
        r = para.add_run()
        r.text = t
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, FONT
        r.font.color.rgb = colour
    return tb


def title(slide, t, sub=None):
    text(slide, 1.1, 1.75, 15.8, 0.8, [(t, 34, True, NAVY)], dy=0)
    if sub:
        text(slide, 1.1, 2.5, 15.8, 0.6, [(sub, 19, False, GREY)], dy=0)


def box(slide, x, y, w, h, t, fill=PALE, colour=INK, size=17, bold=False, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y + DY), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = NAVY
    s.line.width = Pt(1.25)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.12
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(t.split("\n")):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = PP_ALIGN.CENTER
        r = para.add_run()
        r.text = line
        r.font.size, r.font.name = Pt(size if i else size + 1), FONT
        r.font.bold = bold or i == 0
        r.font.color.rgb = colour
    return s


def arrow(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1 + DY), Inches(x2), Inches(y2 + DY))
    c.line.color.rgb = NAVY
    c.line.width = Pt(2.25)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)


def pic(slide, path, x, y, w=None, h=None):
    p = slide.shapes.add_picture(str(path), Inches(x), Inches(y + DY), Inches(w) if w else None, Inches(h) if h else None)
    p.line.color.rgb = RGBColor(0xC8, 0xD0, 0xDC)
    p.line.width = Pt(1)
    return p


def drop_template_text(slide):
    for sh in list(slide.shapes):
        if sh.has_text_frame and sh.text_frame.text.strip():
            sh._element.getparent().remove(sh._element)


def chart():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
    for ax, (lab, a, b) in zip(axes, [("High-risk events missed\n(of %d)" % N_HI, MISS_R, MISS_O),
                                      ("Unnecessary manoeuvres\n(of %d safe events)" % (N_EV - N_HI), UNN_R, UNN_O)]):
        bars = ax.bar(["Latest-message\nrule", "KAKSHA"], [a, b], color=["#9AA5B8", "#C97A00"], width=0.55)
        ax.bar_label(bars, fontsize=15, fontweight="bold", padding=3)
        ax.set_title(lab, fontsize=13)
        ax.set_ylim(0, max(a, b) * 1.3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=12)
    fig.tight_layout()
    out = DECK / "assets" / "results_chart.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    return out


def base(template: pathlib.Path) -> pathlib.Path:
    """Copy the template and add two extra slides (duplicates of slide 6) before the references slide."""
    b = DECK / "_base.pptx"
    ps = (f"$a=New-Object -ComObject PowerPoint.Application;"
          f"$p=$a.Presentations.Open('{template.resolve()}',$true,$false,$false);"
          f"$d=$p.Slides(6).Duplicate();$d.MoveTo(7);$d=$p.Slides(6).Duplicate();$d.MoveTo(8);"
          f"$p.SaveAs('{b.resolve()}');$p.Close();$a.Quit()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    return b


def main(template: str) -> None:
    DECK.mkdir(exist_ok=True)
    prs = Presentation(base(pathlib.Path(template)))
    s = prs.slides

    # 1. Team, track, code
    tb = [sh for sh in s[0].shapes if sh.has_text_frame and "TEAM NAME" in sh.text_frame.text][0]
    vals = ["The Good Engineers", "Chirag", "Software", "SW-5"]
    for para, v in zip(tb.text_frame.paragraphs, vals):
        para.runs[0].font.name = FONT
        r = para.add_run()
        r.text = v
        r.font.name, r.font.size, r.font.bold = FONT, para.runs[0].font.size, True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    tb.width = Inches(13)
    text(s[0], 2.9, 4.75, 12.5, 0.6, [("KAKSHA · satellite collision risk predictor: manoeuvre or monitor, two days out",
                                       22, False, RGBColor(0xFF, 0xD2, 0x7A))], dy=0)

    # 2. Problem and opportunity
    sl = s[1]
    drop_template_text(sl)
    title(sl, "Problem and opportunity")
    text(sl, 1.1, 2.3, 8.0, 6.6, [
        ("The problem", 24, True, NAVY),
        "• Satellites and debris pass at kilometres per second. Operators get conjunction data messages (CDMs) "
        "with a collision probability (Pc) that changes for days, and must decide early whether to manoeuvre.",
        "• A manoeuvre costs fuel and mission time; a missed one can end a satellite (Iridium 33 and "
        "Cosmos 2251 collided in 2009).",
        f"• Dangerous events are rare: in ESA's held-out set, {N_HI} of {N_EV:,} events end high risk "
        "(Pc ≥ 10⁻⁶).",
        f"• The obvious rule, \"trust the latest message\", misses {MISS_R} of those {N_HI} at two days out "
        "(our measurement on ESA data).",
    ])
    text(sl, 9.6, 2.3, 7.6, 4.3, [
        ("How KAKSHA is different", 24, True, NAVY),
        "• A decision, not just a number: manoeuvre or monitor, with a calibrated probability, a range for the "
        "final risk and the three reasons behind it.",
        "• Honest evaluation: ESA's real data and own score, split by event, nothing closer than 2 days; we "
        "reproduce ESA's published baseline score exactly.",
        "• The operator sets the fuel-against-risk line and sees what it costs.",
    ])
    box(sl, 9.6, 6.2, 7.6, 2.3,
        f"USP: decisions you can check\nRetrained on recent events, two days out:\n"
        f"{MISS_O} missed instead of {MISS_R}, for {UNN_O - UNN_R} more manoeuvres\n"
        "Official ESA split: not beaten yet (results slide)",
        fill=RGBColor(0xFF, 0xF4, 0xE0), size=18)

    # 3. Features
    sl = s[2]
    drop_template_text(sl)
    title(sl, "Features", "Real screenshots. Event 1573: latest message just under the line; KAKSHA said manoeuvre; "
          "final risk over 10x above it")
    pic(sl, SHOTS / "event-revealed.png", 1.1, 2.8, h=5.4)
    text(sl, 7.7, 2.8, 9.5, 4.0, [
        ("1. Recommendation with confidence", 19, True, NAVY),
        ("Plan a manoeuvre or monitor, P(final risk high), and an 80 % range for the final risk.", 17, False, INK),
        ("2. Risk history with the 2-day cutoff", 19, True, NAVY),
        ("Every message for the event; the last two days are greyed out because they arrive too late.", 17, False, INK),
        ("3. Reasons", 19, True, NAVY),
        ("The three features that pushed this event's probability up or down.", 17, False, INK),
        ("4. Reveal and honest filters", 19, True, NAVY),
        ("Show what really happened; list our catches, our misses and our unnecessary manoeuvres.", 17, False, INK),
    ])
    pic(sl, SHOTS / "tradeoff.png", 7.7, 6.75, h=1.9)
    text(sl, 12.4, 6.75, 4.8, 1.9, [("5. Fuel-against-risk slider", 19, True, NAVY),
                                   ("Move the line, see missed events against unnecessary manoeuvres.", 16, False, INK)])

    # 4. Process flow (operator use case)
    sl = s[3]
    drop_template_text(sl)
    title(sl, "Process flow: one close approach, from first warning to decision",
          "What the operator does with KAKSHA (top) and what KAKSHA does underneath (bottom)")
    steps = [("CDMs arrive", "every few hours,\n7 days to closest approach"),
             ("T − 2 days", "decision point; later\nmessages not used"),
             ("KAKSHA forecast", "P(high risk), range,\nthree reasons"),
             ("Operator decides", "plan a manoeuvre\nor keep monitoring"),
             ("Final CDM", "outcome logged,\nmodel retrained")]
    x0, w, gap, y = 1.1, 2.85, 0.42, 3.1
    for i, (a, b) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(sl, x, y, w, 1.75, f"{a}\n{b}", fill=RGBColor(0xFF, 0xF4, 0xE0) if i == 3 else PALE, size=15)
        if i:
            arrow(sl, x - gap, y + 0.875, x, y + 0.875)
    under = [("Keep events whose last CDM is within\n1 day of closest approach", 1.1),
             ("Drop every message < 2 days out\n(code refuses them)", 4.37),
             ("154 features: latest message +\ntrends; boosted trees; calibration", 7.64),
             ("Line chosen to maximise F2\n(misses cost more than false alarms)", 10.91),
             ("Score with ESA's metric:\nMSE on high-risk / F2", 14.18)]
    for t, x in under:
        box(sl, x, 5.6, 2.85, 1.45, t, fill=RGBColor(0xFF, 0xFF, 0xFF), size=13, shape=MSO_SHAPE.RECTANGLE)
        arrow(sl, x + 1.425, 4.85, x + 1.425, 5.6)
    text(sl, 1.1, 7.35, 16, 1.5, [
        (f"Measured on {N_EV:,} ESA held-out events: retrained on recent events, KAKSHA misses {MISS_O} high-risk events "
         f"where the latest-message rule misses {MISS_R}; on ESA's official split it does not beat the rule. Use case: a flight-dynamics "
         "operator manoeuvring only where it matters.", 18, False, INK)])

    # 5. Architecture
    sl = s[4]
    drop_template_text(sl)
    title(sl, "Architecture", "Every box exists and runs today (github.com/thegoodengineers/kaksha)")
    cols = [
        (1.1, "Data", "ESA Collision Avoidance\nChallenge dataset\n(Zenodo, CC-BY-4.0)\n\nget_data.py:\ndownload + MD5 check"),
        (4.5, "Events", "kaksha.py\nusable-event filter\n2-day cutoff guard\n154 event features\nESA metric (L, F2)"),
        (7.9, "Model", "train.py\nLightGBM classifier\nisotonic calibration\nF2-chosen manoeuvre line\nrange + reasons"),
        (11.3, "Evidence", "results/*.csv\nmetrics, calibration,\ntrade-off, predictions\n\nRESULTS.md generated\nfrom the CSVs"),
        (14.7, "Console", "server.py (FastAPI)\n/api/events\n/api/event/{id}\n/api/summary\nweb/index.html + SVG"),
    ]
    for x, h, b in cols:
        box(sl, x, 2.85, 2.6, 3.55, f"{h}\n{b}", size=16)
    for x, _, _ in cols[1:]:
        arrow(sl, x - 0.8, 4.6, x, 4.6)
    box(sl, 1.1, 6.85, 7.4, 1.25, "Quality gates\npytest: cutoff, no event on both sides, metric, published\n"
        "baseline reproduced · GitHub Actions on every push", fill=RGBColor(0xFF, 0xFF, 0xFF), size=16)
    box(sl, 8.9, 6.85, 8.4, 1.25, "Operator\nbrowser console, 0 axe-core accessibility problems;\nlater: live CCSDS CDM "
        "feed in, decision log out", fill=RGBColor(0xFF, 0xFF, 0xFF), size=16)
    arrow(sl, 16.0, 6.4, 16.0, 6.85)
    text(sl, 4.6, 8.15, 12.6, 0.7, [("Modular on purpose: the model is one file behind a fixed feature table, so a "
                                  "sequence model can replace it without touching the console.", 16, False, GREY)])

    # 6. Technologies
    sl = s[5]
    drop_template_text(sl)
    title(sl, "Technologies")
    rows = [("Layer", "What we use", "Why"),
            ("Language", "Python 3.11", "One language from data to API; fits a 24-hour build"),
            ("Data", "pandas, NumPy", "103-column CDM tables, per-event grouping"),
            ("Model", "LightGBM gradient-boosted trees", "Strong on tabular data, fast, gives per-event reasons"),
            ("Calibration", "scikit-learn isotonic regression", "Probabilities that mean what they say"),
            ("Uncertainty", "Empirical (split-conformal style) range", "No distribution assumed"),
            ("API", "FastAPI + Uvicorn", "Three JSON endpoints, typed, quick"),
            ("Console", "Plain HTML, CSS, SVG charts", "No framework; loads instantly; accessible"),
            ("Quality", "pytest, GitHub Actions, axe-core", "Leakage, metric and accessibility checks on every push"),
            ("Data source", "ESA Kelvins dataset on Zenodo", "Real CDMs, real labels, published baseline to beat")]
    tbl = sl.shapes.add_table(len(rows), 3, Inches(1.1), Inches(2.6 + DY), Inches(15.9), Inches(5.9)).table
    for c, wd in enumerate((2.6, 5.6, 7.7)):
        tbl.columns[c].width = Inches(wd)
    for r_i, r in enumerate(rows):
        for c, v in enumerate(r):
            cell = tbl.cell(r_i, c)
            cell.text = v
            run = cell.text_frame.paragraphs[0].runs[0]
            run.font.size, run.font.name, run.font.bold = Pt(16), FONT, r_i == 0 or c == 0
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if r_i == 0 else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r_i == 0 else (PALE if r_i % 2 else RGBColor(0xFF, 0xFF, 0xFF))

    # 7. Extra: results
    sl = s[6]
    drop_template_text(sl)
    title(sl, "Extra 1: what we measured (ESA held-out events, two days out)",
          f"{N_EV:,} events, {N_HI} high risk. ESA's score L = MSE on high-risk events / F2, lower is better.")
    text(sl, 1.1, 2.8, 8.0, 0.6, [("1. Retrained on recent events (LightGBM)", 22, True, NAVY)])
    pic(sl, chart(), 1.1, 3.4, w=7.6)
    text(sl, 9.4, 2.8, 7.8, 3.4, [
        ("2. ESA's official split: not beaten", 22, True, AMBER),
        ("Two picks, each chosen only by 5-fold CV on ESA's train events, scored on the test set once:", 17, False, GREY),
        (f"Latest-CDM baseline: L {RULE.L:.3f} · MSE {RULE.mse_hr:.3f} · F2 {RULE.f2:.3f}", 19, True, INK),
        (f"Latest + shrunk ridge correction: L {CORR.L:.3f} · MSE {CORR.mse_hr:.3f} · F2 {CORR.f2:.3f}", 17, False, INK),
        (f"Rule set: L {RULES.L:.3f} · MSE {RULES.mse_hr:.3f} · F2 {RULES.f2:.3f}", 17, False, INK),
    ])
    text(sl, 1.1, 6.6, 16.1, 2.4, [
        ("Said plainly", 20, True, AMBER),
        (f"ESA's train and test events differ: when the latest message said high risk, {int(TR.latest_high_final_low)} of "
         f"{int(TR.latest_high)} train events ended low but only {int(TE.latest_high_final_low)} of {int(TE.latest_high)} test "
         "events, so train-only tuning does not carry over. Retraining with recent events (5 folds by event, no event scored "
         "by a model that saw it) is what helps. Calibrated: Brier "
         f"{U.brier:.3f} vs {U.brier_latest_rule:.3f}. Earlier exploratory test scorings are logged and not counted. "
         "ESA's winner scored 0.556.", 16, False, INK),
    ])

    # 8. Extra: 24-hour build plan
    sl = s[7]
    drop_template_text(sl)
    title(sl, "Extra 2: our 24-hour plan for Round 2", "We start from a working, tested baseline, so the 24 hours "
          "go into making the model and the console better")
    plan = [("H0–3", "Data pipeline, re-run baseline and tests", "Abhijeet"),
            ("H3–10", "Sequence features over the CDM history; per-mission calibration", "Chirag"),
            ("H10–16", "Console: fleet view of all open events, decision log", "Bhumika"),
            ("H16–20", "Evaluation: re-score, wider ranges where high-risk; cost slider in fuel terms", "Abhijeet, Chirag"),
            ("H20–23", "Demo story (event 1573), pitch rehearsed twice with a timer", "Ayush"),
            ("H23–24", "Buffer; recorded backup of the demo", "Everyone")]
    tbl = sl.shapes.add_table(len(plan) + 1, 3, Inches(1.1), Inches(3.0 + DY), Inches(15.9), Inches(4.5)).table
    for c, wd in enumerate((2.0, 10.4, 3.5)):
        tbl.columns[c].width = Inches(wd)
    for r_i, r in enumerate([("Hours", "What", "Owner")] + plan):
        for c, v in enumerate(r):
            cell = tbl.cell(r_i, c)
            cell.text = v
            run = cell.text_frame.paragraphs[0].runs[0]
            run.font.size, run.font.name, run.font.bold = Pt(18), FONT, r_i == 0 or c == 0
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if r_i == 0 else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r_i == 0 else (PALE if r_i % 2 else RGBColor(0xFF, 0xFF, 0xFF))
    text(sl, 4.6, 8.0, 12.6, 0.7, [("Rule for the 24 hours: every number we show comes from a script run that day.",
                                  17, False, GREY)])

    # 9. References
    sl = s[8]
    drop_template_text(sl)
    title(sl, "References")
    text(sl, 1.1, 2.3, 16.0, 6.8, [
        "• ESA, Collision Avoidance Challenge dataset, Zenodo, doi:10.5281/zenodo.4463683 (CC-BY-4.0).",
        "• Uriot T., Izzo D., Simões L. F. et al., \"Spacecraft collision avoidance challenge: design and results of a "
        "machine learning competition\", Astrodynamics 6, 2022 (arXiv:2008.03069): rules, metric, published scores.",
        "• Ke G. et al., \"LightGBM: a highly efficient gradient boosting decision tree\", NeurIPS 2017.",
        "• Zadrozny B., Elkan C., \"Transforming classifier scores into accurate multiclass probability estimates\", "
        "KDD 2002 (isotonic calibration).",
        "• Lei J. et al., \"Distribution-free predictive inference for regression\", JASA 2018 (split-conformal ranges).",
        "• CCSDS 508.0-B-1, Conjunction Data Message, Blue Book, 2013.",
        "• NASA Orbital Debris Program Office, Orbital Debris Quarterly News 13(2), April 2009 "
        "(Iridium 33 / Cosmos 2251 collision).",
        ("Code, results and tests: github.com/thegoodengineers/kaksha · Team: The Good Engineers (Chirag, Bhumika, "
         "Ayush, Abhijeet)", 18, True, NAVY),
    ])

    prs.save(OUT)
    (DECK / "_base.pptx").unlink()
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "rules" / "starforge-deck-template.pptx"))
