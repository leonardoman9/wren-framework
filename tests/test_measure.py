"""The statistics of the bird accuracy measurement (specs/006-bird-accuracy-remeasure, data-model.md), on a fixture of
synthetic windows: no audio, no model.
Usage: .venv-torch/bin/python -m pytest tests/test_measure.py"""
import json

import numpy as np

from wren.torchside.accuracy import evaluate, markdown, measure, recording_of

CLASSES = ["Bubo bubo", "Apus apus", "non-bird"]
INFO = {"classes": CLASSES, "negative": 2, "confidence_threshold": 0.5, "unseen_after": "2025-06-11"}


def fixture():
    """6 species recordings (3 per species) and 2 non-bird ones, 4 windows each; the model is right except on
    recording XC3 and on one non-bird window."""
    names, y = [], []
    recs = [("XC1", 0), ("XC2", 0), ("XC3", 0), ("XC4", 1), ("XC5", 1), ("XC6", 1), ("FS7", 2), ("FS8", 2)]
    for r, c in recs:
        for k in range(4):
            names.append(f"{r}_0.flac@{1.5 * k:.1f}s" if r.startswith("XC") else f"{r}.flac@{3.0 * k:.1f}s"); y.append(c)
    y = np.array(y)
    logits = np.full((len(y), 3), -2.0)
    logits[np.arange(len(y)), y] = 2.0
    for i, n in enumerate(names):
        if n.startswith("XC3"): logits[i] = [-2.0, 2.0, -2.0]   # a whole recording wrong
    logits[names.index("FS8.flac@0.0s")] = [3.0, -2.0, -2.0]   # one confident false detection
    uploaded = {"XC1": "2020-01-01", "XC2": "2020-01-01", "XC3": "2020-01-01", "XC4": "2026-01-01", "XC5": "2020-01-01",
                "XC6": "2020-01-01", "FS7": "2024-01-01", "FS8": "2024-01-01"}
    return y, logits, logits.copy(), names, uploaded


def test_recording_from_window_name():
    assert recording_of("XC803933_2.flac@4.5s") == "XC803933" and recording_of("FS856405.flac@27.0s") == "FS856405"


def test_deterministic_and_interval_contains_value():
    y, f, i, names, up = fixture()
    a, b = measure(y, f, i, names, up, INFO, resamples=300), measure(y, f, i, names, up, INFO, resamples=300)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    acc = a["subsets"]["all"]["float"]["accuracy"]
    assert acc["value"] == round(27 / 32, 4) and acc["ci"][0] <= acc["value"] <= acc["ci"][1]


def test_windows_of_a_recording_move_together():
    """Resampling recordings: every bootstrap accuracy is a mean over whole recordings, so with 8 recordings of 4
    windows each it is a multiple of 1/4 of a recording's share; resampling windows would give other values."""
    y, f, i, names, up = fixture()
    recs = [recording_of(n) for n in names]
    r = evaluate(y, f, i, recs, CLASSES, 2, 0.5, resamples=200)
    lo, hi = r["float"]["accuracy"]["ci"]
    assert lo < 27 / 32 < hi  # the wrong recording is sometimes drawn twice, sometimes not at all


def test_false_detections():
    y, f, i, names, up = fixture()
    a = measure(y, f, i, names, up, INFO, resamples=100)["subsets"]["all"]["int8"]
    assert a["false_detection"]["value"] == round(1 / 8, 4) and a["false_detection_confident"] == round(1 / 8, 4)


def test_minimum_recordings_and_unseen_subset():
    y, f, i, names, up = fixture()
    r = measure(y, f, i, names, up, INFO, resamples=100)
    assert all(not v["meaningful"] for v in r["per_class"].values())  # 3 or 2 recordings each, fewer than 5
    assert r["subsets"]["unseen"]["recordings"] == 1  # XC4, uploaded after 2025-06-11
    none_unseen = {k: "2020-01-01" for k in up}
    assert measure(y, f, i, names, none_unseen, INFO, resamples=100)["subsets"]["unseen"] is None
    assert "not meaningful" in markdown(r)


def test_rounded_to_four_decimals():
    y, f, i, names, up = fixture()
    text = json.dumps(measure(y, f, i, names, up, INFO, resamples=100))
    assert not any(len(t.split(".")[1]) > 4 for t in text.replace(",", " ").replace("]", " ").split() if t[:1].isdigit() and "." in t)
