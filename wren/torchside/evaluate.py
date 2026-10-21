"""Stage evaluate (torch environment): the trained model on the test split, measured the way the device uses it.
Usage: python -m wren.torchside.evaluate SPEC.json PREPARE_DIR CHECKPOINT.pt OUT_DIR
SPEC.json: classes, negative, device_hop_seconds, device_confidence.

(a) call windows: the windows prepare selected (one label each): accuracy, macro-F1, per class, confusion matrix.
(b) device decision per file: every window on the device's grid (hop device_hop_seconds over the file), the device's
    rule (top class, rounded percentage >= threshold, not the negative class) -> the classes the device would report
    for the file. Per class: precision and recall over files, for thresholds 30-90 %. The threshold with the best file
    macro-F1 on the VALIDATION split is suggested; the test split is reported at it (and at the configured one).
(c) false reports per hour on negative files (only background), with the raw count and the hours of audio.
Gate: macro-F1 of (a) must exceed 1/N + 0.1, otherwise the data or the pipeline is broken.
Outputs: evaluate.json, confusion.csv. Note: a file shorter than 3 s gives no window on the device (here it is looped).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from wren.torchside.metrics import scores
from wren.torchside.model import Core, device_forward, load_model

WIN, SR = 96000, 32000
spec = json.loads(Path(sys.argv[1]).read_text())
prep, ckpt, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
classes, neg = spec["classes"], spec["negative"]
K = len(classes)
m = load_model(ckpt)
core = Core(m).eval()
def load_split(name):
    files = json.loads((prep / f"files_{name}.json").read_text())
    windows = json.loads((prep / f"windows_{name}.json").read_text())
    audio = np.memmap(prep / f"audio_{name}.raw", dtype=np.int16, mode="r", shape=(sum(f["length"] for f in files),))
    return files, windows, audio


def probs(items, files, audio):
    """Softmax for (file index, start) pairs, in batches."""
    res = []
    with torch.no_grad():
        for i in range(0, len(items), 64):
            x = np.stack([audio[files[fi]["offset"] + s: files[fi]["offset"] + s + WIN].astype(np.float32) / 32768.0 for fi, s in items[i:i + 64]])
            res.append(torch.softmax(device_forward(m, core, torch.from_numpy(x)), dim=1).numpy())
    return np.concatenate(res) if res else np.zeros((0, K))


hop = int(round(spec["device_hop_seconds"] * SR))
positives = [c for c in range(K) if c != neg]


def device_grid(files, audio):
    """Per file, the softmax of every window on the device's grid."""
    grid = [(fi, s) for fi, f in enumerate(files) for s in (range(0, f["length"] - WIN + 1, hop) or [0])]
    per_file = {}
    for (fi, _), p in zip(grid, probs(grid, files, audio)): per_file.setdefault(fi, []).append(p)
    return per_file, len(grid)


def at_threshold(pct, files, per_file):
    t = (pct - 0.5) / 100.0  # the device rounds the percentage (Sched_TopPercent) before comparing
    reported = {fi: {int(p.argmax()) for p in ps if p.max() >= t and int(p.argmax()) != neg} for fi, ps in per_file.items()}
    cls = {}
    for c in positives:
        tp = sum(1 for fi in reported if c in reported[fi] and files[fi]["label"] == c)
        fp = sum(1 for fi in reported if c in reported[fi] and files[fi]["label"] != c)
        fn = sum(1 for fi in reported if c not in reported[fi] and files[fi]["label"] == c)
        p_, r_ = tp / max(1, tp + fp), tp / max(1, tp + fn)
        cls[classes[c]] = {"precision": round(p_, 4), "recall": round(r_, 4), "f1": round(2 * p_ * r_ / max(1e-9, p_ + r_), 4)}
    neg_files = [fi for fi in per_file if files[fi]["label"] == neg]
    hours = sum(files[fi]["length"] for fi in neg_files) / SR / 3600
    false_reports = sum(1 for fi in neg_files for p in per_file[fi] if p.max() >= t and int(p.argmax()) != neg)
    return {"confidence_pct": pct, "macro_f1": round(float(np.mean([v["f1"] for v in cls.values()])) if cls else 0.0, 4),
            "per_class": cls, "false_reports": false_reports, "background_hours": round(hours, 4),
            "false_reports_per_hour": round(false_reports / hours, 2) if hours > 0 else None}


# (a) on the test call windows
files, windows, audio = load_split("test")
pa = probs([(fi, s) for fi, s, _ in windows], files, audio)
a = scores([l for _, _, l in windows], pa.argmax(1).tolist(), K, classes)
# (b), (c): the threshold is chosen on validation, the test is reported at it
vfiles, _, vaudio = load_split("validation")
vper, _ = device_grid(vfiles, vaudio)
pcts = list(range(30, 91, 10))
vsweep = [at_threshold(c, vfiles, vper) for c in pcts]
suggested = max(vsweep, key=lambda r: (r["macro_f1"], r["confidence_pct"]))["confidence_pct"]
per_file, n_grid = device_grid(files, audio)
result = {"call_windows": {k: a[k] for k in ("accuracy", "macro_f1", "per_class")}, "test_windows": len(windows),
          "test_files": len(files), "device_grid_windows": n_grid,
          "files": {"suggested_confidence_pct": suggested, "chosen_on": "validation",
                    "at_suggested": at_threshold(suggested, files, per_file),
                    "at_configured": at_threshold(spec["device_confidence"], files, per_file),
                    "test_sweep": [at_threshold(c, files, per_file) for c in pcts], "validation_sweep": vsweep},
          "chance_macro_f1": round(1.0 / K, 4)}
(out / "evaluate.json").write_text(json.dumps(result, indent=2) + "\n")
with open(out / "confusion.csv", "w") as f:
    f.write("true\\predicted," + ",".join(classes) + "\n")
    for i, row in enumerate(a["confusion"]): f.write(classes[i] + "," + ",".join(map(str, row)) + "\n")
best = result["files"]["at_suggested"]
print(json.dumps({"accuracy": a["accuracy"], "macro_f1": a["macro_f1"], "suggested_confidence_pct": suggested,
                  "test_file_macro_f1_at_suggested": best["macro_f1"], "false_reports_per_hour": best["false_reports_per_hour"]}))
if a["macro_f1"] <= 1.0 / K + 0.1:
    sys.exit(f"evaluate: macro-F1 {a['macro_f1']:.3f} is at chance level (1/{K} = {1 / K:.3f}): data or pipeline broken")
