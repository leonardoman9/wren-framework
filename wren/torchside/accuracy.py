"""The bird model's accuracy on the test bundle (torch environment, numpy only; specs/006-bird-accuracy-remeasure).
Usage: python -m wren.torchside.accuracy WINDOWS.npz REF.npz ATTRIBUTION.csv INFO.json OUT_DIR

WINDOWS.npz: labels, float_logits, int8_logits per window (wren/tfside/device_eval.py); REF.npz: the windows' names
(`XC<id>_<k>.flac@<s>s`, `FS<id>.flac@<s>s`: the recording is the prefix); ATTRIBUTION.csv: `uploaded` per recording;
INFO.json: classes (model order), negative, confidence threshold, unseen_after, and the hashes to record. Writes
accuracy.json (sorted keys, 4 decimals) and accuracy.md. Intervals: 95 % percentile bootstrap over recordings (the
windows of a recording move together), 2,000 resamples, seed 0, so the output is the same on every run."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

RESAMPLES, SEED, LEVEL, MIN_RECORDINGS = 2000, 0, 0.95, 5


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def recording_of(name):
    return re.match(r"(XC\d+|FS\d+)", name).group(1)


def scores(y, pred, k):
    present = [c for c in range(k) if (y == c).any()]
    recall, precision, f1 = {}, {}, []
    for c in present:
        tp = int(((pred == c) & (y == c)).sum())
        r = tp / int((y == c).sum()); p = tp / max(1, int((pred == c).sum()))
        recall[c], precision[c] = r, p
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float((pred == y).mean()), float(np.mean(f1)), recall, precision


def false_detections(y, prob, negative, threshold):
    bg = y == negative
    if negative < 0 or not bg.any(): return None, None
    pred = prob.argmax(1)
    wrong = pred[bg] != negative
    return float(wrong.mean()), float((wrong & (prob[bg].max(1) >= threshold)).mean())


def interval(values):
    lo, hi = np.percentile(values, [100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2])
    return [float(lo), float(hi)]


def evaluate(y, flog, ilog, recordings, classes, negative, threshold, resamples=RESAMPLES, seed=SEED):
    """Point values and recording-bootstrap intervals for one set of windows (data-model.md, one subset)."""
    k = len(classes)
    probs = {"float": softmax(flog), "int8": softmax(ilog)}
    recs = sorted(set(recordings))
    rows_of = {r: np.flatnonzero(np.array(recordings) == r) for r in recs}
    rng = np.random.default_rng(seed)
    boot = {m: {"accuracy": [], "macro_f1": [], "false_detection": []} for m in probs}
    diff = []
    for _ in range(resamples):
        idx = np.concatenate([rows_of[recs[i]] for i in rng.integers(0, len(recs), len(recs))])
        mf = {}
        for m, p in probs.items():
            acc, f1, _, _ = scores(y[idx], p[idx].argmax(1), k)
            boot[m]["accuracy"].append(acc); boot[m]["macro_f1"].append(f1); mf[m] = f1
            fd, _ = false_detections(y[idx], p[idx], negative, threshold)
            if fd is not None: boot[m]["false_detection"].append(fd)
        diff.append(mf["int8"] - mf["float"])
    res = {}
    for m, p in probs.items():
        pred = p.argmax(1)
        acc, f1, recall, precision = scores(y, pred, k)
        fd, fdc = false_detections(y, p, negative, threshold)
        conf = np.zeros((k, k), dtype=int)
        for t, q in zip(y, pred): conf[t, q] += 1
        res[m] = {"accuracy": {"value": acc, "ci": interval(boot[m]["accuracy"])},
                  "macro_f1": {"value": f1, "ci": interval(boot[m]["macro_f1"])},
                  "recall": {classes[c]: v for c, v in recall.items()},
                  "precision": {classes[c]: v for c, v in precision.items()},
                  "confusion": conf.tolist(),
                  "false_detection": None if fd is None else {"value": fd, "ci": interval(boot[m]["false_detection"])},
                  "false_detection_confident": fdc}
    _, f1f, _, _ = scores(y, probs["float"].argmax(1), k)
    _, f1i, _, _ = scores(y, probs["int8"].argmax(1), k)
    res["int8_minus_float_macro_f1"] = {"value": f1i - f1f, "ci": interval(diff)}
    res["windows"], res["recordings"] = int(len(y)), len(recs)
    return res


def rounded(x):
    if isinstance(x, float): return round(x, 4)
    if isinstance(x, dict): return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list): return [rounded(v) for v in x]
    return x


def measure(y, flog, ilog, names, uploaded, info, resamples=RESAMPLES, seed=SEED):
    classes, negative, threshold = info["classes"], info["negative"], info["confidence_threshold"]
    recordings = [recording_of(n) for n in names]
    per_class = {}
    for c, name in enumerate(classes):
        rs = {r for r, l in zip(recordings, y) if l == c}
        per_class[name] = {"recordings": len(rs), "windows": int((y == c).sum()), "meaningful": len(rs) >= MIN_RECORDINGS}
    unseen = np.array([r.startswith("XC") and uploaded.get(r, "") > info["unseen_after"] for r in recordings])
    subsets = {"all": evaluate(y, flog, ilog, recordings, classes, negative, threshold, resamples, seed)}
    subsets["unseen"] = (evaluate(y[unseen], flog[unseen], ilog[unseen], [r for r, u in zip(recordings, unseen) if u],
                                  classes, negative, threshold, resamples, seed) if unseen.any() else None)
    out = {k: v for k, v in info.items()}
    out.update(min_recordings=MIN_RECORDINGS, bootstrap={"resamples": resamples, "seed": seed, "level": LEVEL},
               per_class=per_class, subsets=subsets)
    return rounded(out)


def pct(v):
    return "–" if v is None else f"{100 * v:.1f} %"


def ci(d):
    return "–" if d is None else f"{100 * d['value']:.1f} % ({100 * d['ci'][0]:.1f}–{100 * d['ci'][1]:.1f})"


def markdown(r):
    a = r["subsets"]["all"]
    lines = ["# Bird model accuracy on the test bundle", "",
             f"{a['windows']} windows from {a['recordings']} recordings (`examples/birds/test/`); 95 % intervals by bootstrap "
             f"over recordings. Device = C front-end + int8 backbone + C tail, the firmware's own code.", "",
             "| | float (PyTorch) | device (int8) |", "|---|---|---|",
             f"| accuracy | {ci(a['float']['accuracy'])} | {ci(a['int8']['accuracy'])} |",
             f"| macro-F1 | {ci(a['float']['macro_f1'])} | {ci(a['int8']['macro_f1'])} |",
             f"| non-bird windows called a species | {ci(a['float']['false_detection'])} | {ci(a['int8']['false_detection'])} |",
             f"| ... with confidence ≥ {100 * r['confidence_threshold']:.0f} % | {pct(a['float']['false_detection_confident'])} | "
             f"{pct(a['int8']['false_detection_confident'])} |",
             f"| device − float, macro-F1 | | {100 * a['int8_minus_float_macro_f1']['value']:+.1f} points "
             f"({100 * a['int8_minus_float_macro_f1']['ci'][0]:+.1f} to {100 * a['int8_minus_float_macro_f1']['ci'][1]:+.1f}) |", "",
             "| class | recordings | windows | recall (device) | precision (device) |", "|---|---|---|---|---|"]
    for c, v in r["per_class"].items():
        note = "" if v["meaningful"] else f" (fewer than {r['min_recordings']} recordings: not meaningful)"
        lines.append(f"| {c}{note} | {v['recordings']} | {v['windows']} | {pct(a['int8']['recall'].get(c))} | "
                     f"{pct(a['int8']['precision'].get(c))} |")
    u = r["subsets"]["unseen"]
    lines += ["", f"Recordings uploaded after {r['unseen_after']} (certainly not in WrenNet's training data): " +
              ("none in the bundle." if u is None else
               f"{u['windows']} windows from {u['recordings']} recordings, accuracy {ci(u['float']['accuracy'])} (float), "
               f"{ci(u['int8']['accuracy'])} (device).")]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    win, ref, attribution, info_path, out = (Path(a) for a in sys.argv[1:6])
    w, names = np.load(win), [str(n) for n in np.load(ref)["names"]]
    with attribution.open() as f:
        uploaded = {("XC" if r["source"] == "xeno-canto" else "FS") + r["source_id"]: r["uploaded"] for r in csv.DictReader(f)}
    result = measure(w["labels"], w["float_logits"], w["int8_logits"], names, uploaded, json.loads(info_path.read_text()))
    out.mkdir(parents=True, exist_ok=True)
    (out / "accuracy.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "accuracy.md").write_text(markdown(result))
    print(markdown(result))
