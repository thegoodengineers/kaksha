"""KAKSHA operator console: FastAPI over results/ (from train.py) and data/ (from scripts/get_data.py).

Run: python -m uvicorn server:app --port 8780
"""
from __future__ import annotations

import pathlib

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import kaksha as k

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results"

pred = pd.read_csv(RES / "predictions.csv", index_col="event_id")
cdms = k.load("test_data")[["event_id", "time_to_tca", "risk", "miss_distance", "max_risk_estimate"]]
final = k.load("test_data_private").set_index("event_id")[["time_to_tca"]]


def table(name: str) -> list[dict]:
    return pd.read_csv(RES / f"{name}.csv").to_dict(orient="records")


def kind(r) -> str:
    """How this event went, judged against what really happened (shown only after 'reveal')."""
    high = r.true_final_risk >= k.HIGH
    rule = r.latest_risk >= k.HIGH
    if high and r.manoeuvre and not rule:
        return "caught: latest risk said low"
    if high and not r.manoeuvre:
        return "missed"
    if not high and r.manoeuvre:
        return "unnecessary manoeuvre"
    if high:
        return "correct manoeuvre"
    return "correct monitor"


app = FastAPI(title="KAKSHA")


@app.get("/api/summary")
def summary():
    return {"metrics": table("metrics"), "tradeoff": table("tradeoff"),
            "calibration": table("calibration"), "data": table("data_summary"),
            "uncertainty": table("uncertainty")}


@app.get("/api/events")
def events():
    out = []
    for eid, r in pred.iterrows():
        out.append({"id": int(eid), "mission": str(r.mission_id), "latest": round(r.latest_risk, 2),
                    "p": round(r.p_high, 3), "manoeuvre": bool(r.manoeuvre), "kind": kind(r)})
    return sorted(out, key=lambda e: -e["p"])


@app.get("/api/event/{eid}")
def event(eid: int):
    if eid not in pred.index:
        raise HTTPException(404, "no such event in the held-out set")
    r = pred.loc[eid]
    tl = cdms[cdms.event_id == eid].sort_values("time_to_tca", ascending=False)
    return {
        "id": eid, "mission": str(r.mission_id), "n_cdms": int(r.n_cdms),
        "timeline": tl.drop(columns="event_id").round(4).to_dict(orient="records"),
        "p_high": r.p_high, "threshold": r.threshold, "manoeuvre": bool(r.manoeuvre),
        "q10": r.q10, "q50": r.q50, "q90": r.q90, "latest": r.latest_risk,
        "reasons": r.reasons.split("; "),
        "outcome": {"final_risk": r.true_final_risk, "time_to_tca": float(final.loc[eid, "time_to_tca"]),
                    "kind": kind(r)},
    }


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")
