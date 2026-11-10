"""Checks on the 2-day cutoff, the event split (no event on both sides) and the metric code."""
import numpy as np
import pandas as pd
import pytest

import kaksha as k
import train


def cdm(event, t, risk):
    row = {c: 1.0 for c in k.TREND_COLS}
    return {**row, "event_id": event, "time_to_tca": t, "risk": risk, "mission_id": 1, "c_object_type": "DEBRIS"}


def toy():
    return pd.DataFrame([
        cdm(1, 5.0, -7), cdm(1, 3.0, -6.5), cdm(1, 1.5, -5), cdm(1, 0.5, -4.2),  # kept, final -4.2
        cdm(2, 4.0, -8), cdm(2, 1.2, -8),                                       # last CDM 1.2 d out: dropped
        cdm(3, 1.8, -5), cdm(3, 0.3, -5),                                       # nothing 2+ days out: dropped
        cdm(4, 2.0, -30), cdm(4, 0.9, -30),                                     # kept, exactly 2 d is allowed
    ])


def test_split_keeps_only_usable_events_and_final_label():
    early, final = k.split_cdms(toy())
    assert sorted(final.index) == [1, 4]
    assert final[1] == -4.2 and final[4] == -30
    assert (early.time_to_tca >= k.CUTOFF).all()
    assert sorted(early.event_id.unique()) == [1, 4]


def test_features_use_only_cdms_two_days_out():
    early, _ = k.split_cdms(toy())
    X = k.features(early)
    assert X.loc[1, "last_risk"] == -6.5  # the -5 message 1.5 days out is never seen
    assert X.loc[1, "n_cdms"] == 2 and X.loc[1, "risk_change"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        k.features(toy())


def test_folds_never_put_an_event_on_both_sides():
    rng = np.random.default_rng(0)
    events = np.arange(500)
    y = np.where(rng.random(500) < 0.07, -5.0, -9.0)
    seen = []
    for a, b in train.folds(y):
        assert not set(events[a]) & set(events[b])
        seen.extend(events[b])
    assert sorted(seen) == list(events)  # every event scored exactly once


def test_fbeta_matches_hand_count():
    t = np.array([1, 1, 1, 0, 0], bool)
    p = np.array([1, 1, 0, 1, 0], bool)  # tp 2, fp 1, fn 1 -> P = R = 2/3 -> F2 = 2/3
    assert k.fbeta(t, p) == pytest.approx(2 / 3)
    assert k.fbeta(t, np.zeros(5, bool)) == 0.0


def test_score_hand_example():
    true = np.array([-5.0, -4.0, -8.0, -30.0])
    pred = k.clip(np.array([-5.5, -7.0, -5.0, -12.0]))  # -7 and -12 become -6.001
    s = k.score(true, pred)
    assert s["high_risk_events"] == 2
    assert s["mse_hr"] == pytest.approx((0.5**2 + 2.001**2) / 2)
    # tp 1, fp 1, fn 1 -> F2 = 0.5
    assert s["f2"] == pytest.approx(0.5)
    assert s["L"] == pytest.approx(s["mse_hr"] / 0.5)
    assert s["missed_high_risk"] == 1 and s["unnecessary_manoeuvres"] == 1
    assert s["false_alarm_rate"] == pytest.approx(0.5)


@pytest.mark.skipif(not (k.DATA / "test_data.csv").exists(), reason="run scripts/get_data.py first")
def test_reproduces_published_baseline():
    """Uriot et al. 2022 report L = 0.694 for 'latest CDM risk' on 2,167 test events, 150 high risk."""
    X = k.features(k.load("test_data"))
    truth = k.load("test_data_private").set_index("event_id").true_risk.reindex(X.index)
    s = k.score(truth, k.clip(X.last_risk))
    assert s["events"] == 2167 and s["high_risk_events"] == 150
    assert round(s["L"], 3) == 0.694


@pytest.mark.skipif(not (k.DATA / "train_data.csv").exists(), reason="run scripts/get_data.py first")
def test_official_train_and_test_events_are_disjoint_by_content():
    """ESA's train and test events reuse event_id numbers, so compare each event's first CDM on
    every numeric column (four columns alone collide by chance for one repeating mission)."""
    def first(d):
        d = d.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).groupby("event_id").head(1)
        return set(map(tuple, d.drop(columns=["event_id", "c_object_type"]).round(6).astype(str).to_numpy()))
    assert not first(k.load("train_data")) & first(k.load("test_data"))
