"""Unit checks of the training side (prepare, model, windows, error paths). Run with the torch environment:
  .venv-torch/bin/python tests/test_training.py
Prints PASS/FAIL per check; exits 1 if any failed. Needs no data beyond what it makes in a temporary folder."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wren import config  # noqa: E402
from wren.torchside.model import Core, build_model, device_forward, load_model  # noqa: E402

PY = sys.executable
SR, WIN = 32000, 96000
results = []


def check(name, ok, detail=""):
    results.append(ok); print(("PASS " if ok else "FAIL ") + name + (f" | {detail}" if detail else ""), flush=True)


def tone(seconds, events, f=4000.0, level=0.3, seed=0):
    """Noise with tonal events (start seconds) of 0.4 s."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, int(seconds * SR)).astype(np.float32)
    t = np.arange(int(0.4 * SR)) / SR
    for e in events:
        s = int(e * SR); x[s:s + len(t)] += (level * np.sin(2 * np.pi * f * t) * np.hanning(len(t))).astype(np.float32)
    return x


def write(path, x, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True); soundfile.write(str(path), x, sr, subtype="PCM_16")


def prepare(root, spec_extra, out, expect_ok=True):
    spec = {"classes": sorted(d.name for d in (root / "train" if (root / "train").exists() else root).iterdir()), "negative": -1,
            "sources": {"root": str(root)}, "split": {"validation": 0.15, "test": 0.15, "seed": 0}, "group": "file",
            "windows": {"hop_seconds": 1.5, "max_per_file": 6, "negative_max_per_file": 40}, "device_hop_seconds": 3.0, **spec_extra}
    f = out.parent / (out.name + "_spec.json"); f.write_text(json.dumps(spec))
    r = subprocess.run([PY, "-m", "wren.torchside.prepare", f, out], capture_output=True, text=True, cwd=ROOT)
    return r


tmp = Path(tempfile.mkdtemp(prefix="wren_test_"))
try:
    # --- model ---------------------------------------------------------------------------------------------------
    m = load_model(ROOT / "wrennet/best_distillation_model.pt"); core = Core(m).eval()
    rng = np.random.default_rng(0)
    x = torch.from_numpy(np.stack([tone(3, [0.5], level=l, seed=i) for i, l in enumerate([0.001, 0.05, 0.3, 0.9])]))
    with torch.no_grad():
        batched = device_forward(m, core, x)
        single = torch.cat([device_forward(m, core, x[i:i + 1]) for i in range(len(x))])
        model_single = torch.cat([m(x[i:i + 1]) for i in range(len(x))])
    check("batched = one clip at a time", float((batched - single).abs().max()) < 1e-4, f"max |d| {float((batched - single).abs().max()):.2e}")
    check("device_forward = WrenNet model on one clip", float((single - model_single).abs().max()) < 1e-5)
    s = build_model(7)
    with torch.no_grad(): out = device_forward(s, Core(s), x)
    check("model from scratch: finite filterbank and logits", bool(torch.isfinite(out).all()) and out.shape == (4, 7))

    # --- prepare: windows on long files with events at known times -------------------------------------------------
    root = tmp / "data"
    events = {"bird": [2.0, 9.0, 17.5], "frog": [5.0, 6.0, 14.0]}  # frog: two events 1 s apart (one window)
    for c, ev in events.items():
        for i in range(6):
            write(root / c / f"{c}_{i}.WAV", tone(24, ev, f=3000 if c == "bird" else 1500, seed=i))  # upper-case extension
    for i in range(6): write(root / "noise" / f"n_{i}.flac", tone(24, [], seed=100 + i))
    r = prepare(root, {"classes": ["bird", "frog", "noise"], "negative": 2}, tmp / "prep1")
    check("prepare runs (.WAV and .flac found)", r.returncode == 0, r.stderr[-900:])
    if r.returncode == 0:
        files = {s_: json.loads((tmp / "prep1" / f"files_{s_}.json").read_text()) for s_ in ("train", "validation", "test")}
        wins = {s_: json.loads((tmp / "prep1" / f"windows_{s_}.json").read_text()) for s_ in ("train", "validation", "test")}
        ok_found, ok_apart, ok_count = True, True, True
        for s_ in files:
            for fi, f in enumerate(files[s_]):
                if f["class"] == "noise": continue
                starts = sorted(w[1] / SR for w in wins[s_] if w[0] == fi)
                ev = events[f["class"]]
                ok_found &= all(any(st <= e and e + 0.4 <= st + 3.0 for st in starts) for e in ev)  # every event inside a window
                ok_apart &= all(b - a >= 3.0 for a, b in zip(starts, starts[1:]))
                ok_count &= len(starts) <= (3 if f["class"] == "bird" else 2)  # no windows without an event
        check("every event inside a selected window", ok_found)
        check("selected windows at least 3 s apart", ok_apart)
        check("no extra windows without events", ok_count)
        paths = {s_: {f["path"] for f in files[s_]} for s_ in files}
        check("splits disjoint by file", not (paths["train"] & paths["test"] or paths["train"] & paths["validation"] or paths["validation"] & paths["test"]))
        r2 = prepare(root, {"classes": ["bird", "frog", "noise"], "negative": 2}, tmp / "prep2")
        same = all((tmp / "prep1" / f"{k}_{s_}.json").read_text() == (tmp / "prep2" / f"{k}_{s_}.json").read_text()
                   for k in ("files", "windows") for s_ in files)
        same &= all((tmp / "prep1" / f"audio_{s_}.raw").read_bytes() == (tmp / "prep2" / f"audio_{s_}.raw").read_bytes() for s_ in files)
        check("prepare is deterministic", r2.returncode == 0 and same)
        neg = [w for s_ in wins for w in wins[s_] if w[2] == 2]
        check("negative windows on the device grid", all(w[1] % (3 * SR) == 0 for w in neg) and len(neg) > 0)

    # --- prepare: a weak call among loud low-frequency gusts; a file with only noise ------------------------------
    def gusty(seconds, calls, gusts, seed):
        rng_ = np.random.default_rng(seed)
        x = rng_.normal(0, 0.003, int(seconds * SR)).astype(np.float32)
        spec_ = np.fft.rfft(rng_.normal(0, 1, len(x))); spec_[np.fft.rfftfreq(len(x), 1 / SR) > 300] = 0
        lowpass = np.fft.irfft(spec_, n=len(x)).astype(np.float32)  # wind: energy below 300 Hz only
        for g in gusts:  # wind: loud, 2 s, low frequency
            s0 = int(g * SR); n = 2 * SR; x[s0:s0 + n] += 0.5 * lowpass[s0:s0 + n] / (np.std(lowpass) + 1e-9) * np.hanning(n)
        t = np.arange(int(0.4 * SR)) / SR
        for c in calls:  # a weak 6 kHz call
            s0 = int(c * SR); x[s0:s0 + len(t)] += (0.02 * np.sin(2 * np.pi * 6000 * t) * np.hanning(len(t))).astype(np.float32)
        return x
    root_g = tmp / "gusty"
    # wind as loud and as frequent in every class (it does not depend on the animal); 2 "bird" files without a bird
    for i in range(8):
        write(root_g / "bird" / f"b{i}.wav", gusty(30, [4.0, 21.0], [10.0, 15.0, 26.0], seed=i))
        write(root_g / "noise" / f"n{i}.wav", gusty(30, [], [6.0, 12.0, 24.0], seed=100 + i))
    for i in range(2): write(root_g / "bird" / f"silent{i}.wav", gusty(30, [], [8.0, 18.0, 25.0], seed=50 + i))
    r = prepare(root_g, {"classes": ["bird", "noise"], "negative": 1}, tmp / "prep_gusty")
    ok_calls, ok_silent, detail = r.returncode == 0, True, r.stderr[-200:]
    if r.returncode == 0:
        for s_ in ("train", "validation", "test"):
            files = json.loads((tmp / "prep_gusty" / f"files_{s_}.json").read_text())
            wins = json.loads((tmp / "prep_gusty" / f"windows_{s_}.json").read_text())
            for fi, f in enumerate(files):
                if f["class"] != "bird": continue
                starts = [w[1] / SR for w in wins if w[0] == fi]
                if "silent" in f["path"]: ok_silent &= not starts
                else: ok_calls &= len(starts) == 2 and all(any(st <= c and c + 0.4 <= st + 3 for st in starts) for c in (4.0, 21.0))
        detail = f"selected starts of one file with calls: {[w[1] / SR for w in wins if w[0] == 0]}"
        band = json.loads((tmp / "prep_gusty" / "prepare.json").read_text())["class_bands_hz"].get("bird")
        ok_calls &= band is not None and band[0] <= 6000 <= band[1]; detail += f", band {band}"
    check("weak calls found despite loud wind gusts", ok_calls, detail)
    check("a long file with only noise gives no windows", ok_silent)

    # wind only in the animal's recordings: cannot be told apart from the animal, but must be reported
    root_w = tmp / "wind_one_class"
    for i in range(8):
        write(root_w / "bird" / f"b{i}.wav", gusty(30, [4.0, 21.0], [10.0, 15.0, 26.0], seed=i))
        write(root_w / "noise" / f"n{i}.wav", gusty(30, [], [], seed=100 + i))
    r = prepare(root_w, {"classes": ["bird", "noise"], "negative": 1}, tmp / "prep_wind")
    warned = r.returncode == 0 and any("bird" in w and ("Hz" in w) for w in json.loads((tmp / "prep_wind" / "prepare.json").read_text())["warnings"])
    check("noise in one class only: reported", warned, r.stderr[-200:])

    # a chorus: the animal calls in most frames of every clip (the floor must not become the call)
    root_c = tmp / "chorus"
    for i in range(6):
        rng_ = np.random.default_rng(200 + i)
        x = rng_.normal(0, 0.01, 10 * SR).astype(np.float32)
        t = np.arange(10 * SR) / SR
        x += (0.1 * np.sin(2 * np.pi * 1500 * t) * ((t * 5 + rng_.uniform()) % 1 < 0.6)).astype(np.float32)  # 60 % on
        write(root_c / "frog" / f"f{i}.wav", x)
        write(root_c / "bird" / f"b{i}.wav", tone(10, [2.0, 6.0], f=5000, seed=300 + i))
        write(root_c / "noise" / f"n{i}.wav", tone(10, [], seed=400 + i))
    r = prepare(root_c, {"classes": ["bird", "frog", "noise"], "negative": 2}, tmp / "prep_chorus")
    ok = r.returncode == 0
    if ok:
        for s_ in ("train", "validation", "test"):
            files = json.loads((tmp / "prep_chorus" / f"files_{s_}.json").read_text())
            wins = json.loads((tmp / "prep_chorus" / f"windows_{s_}.json").read_text())
            ok &= all(any(w[0] == fi for w in wins) for fi, f in enumerate(files) if f["class"] == "frog")
    check("a chorus (calls in 60 % of the frames) gives windows", ok, r.stderr[-200:])

    # --- prepare: short files are looped, not padded with silence ---------------------------------------------------
    root2 = tmp / "short"
    for c in ("a", "b"):
        for i in range(6): write(root2 / c / f"{c}{i}.wav", tone(1.2, [0.3], f=2000 if c == "a" else 5000, seed=i), sr=24000)
    r = prepare(root2, {"classes": ["a", "b"]}, tmp / "prep_short")
    ok = r.returncode == 0
    if ok:
        f0 = json.loads((tmp / "prep_short" / "files_train.json").read_text())[0]
        a = np.fromfile(tmp / "prep_short" / "audio_train.raw", dtype=np.int16)[f0["offset"]: f0["offset"] + f0["length"]]
        ok = f0["length"] == WIN and f0["padded"] and np.abs(a[-SR // 2:]).max() > 0  # no digital silence at the end
        rep = json.loads((tmp / "prep_short" / "prepare.json").read_text())
        ok &= any("24000 Hz" in w for w in rep["warnings"])
    check("short files looped to 3 s; low sample rate reported", ok, r.stderr[-200:])

    # --- errors ---------------------------------------------------------------------------------------------------
    root3 = tmp / "few"
    for c, n in (("a", 6), ("b", 3)):
        for i in range(n): write(root3 / c / f"{c}{i}.wav", tone(3, [1], seed=i))
    r = prepare(root3, {"classes": ["a", "b"]}, tmp / "prep_few")
    check("error: a class with fewer than 5 files", r.returncode != 0 and "at least 5" in r.stderr, r.stderr[-150:])
    shutil.copy(root / "bird" / "bird_0.WAV", root / "frog" / "copy_of_bird.wav")
    r = prepare(root, {"classes": ["bird", "frog", "noise"], "negative": 2}, tmp / "prep_dup")
    check("error: the same recording in two classes", r.returncode != 0 and "duplicate" in r.stderr, r.stderr[-150:])
    (root / "frog" / "copy_of_bird.wav").unlink()
    split_root = tmp / "split"
    for s_, cls in (("train", ["a", "b"]), ("validation", ["a", "c"])):
        for c in cls:
            for i in range(6): write(split_root / s_ / c / f"{c}{i}_{s_}.wav", tone(3, [1], seed=i + (50 if s_ == "validation" else 0)))
    spec = {"classes": ["a", "b"], "negative": -1, "sources": {"train": str(split_root / "train"), "validation": str(split_root / "validation")}}
    r = prepare(split_root / "train", spec, tmp / "prep_split")
    check("error: split folders with different classes", r.returncode != 0 and "differ" in r.stderr, r.stderr[-150:])
    yml = tmp / "bad.yaml"
    (tmp / "bad" / "ok_class").mkdir(parents=True); (tmp / "bad" / "bad,name").mkdir()
    yml.write_text(f"name: bad\ndata: {{root: {tmp / 'bad'}}}\n")
    try:
        config.load(yml); check("error: class name the firmware cannot store", False)
    except SystemExit as e:
        check("error: class name the firmware cannot store", "bad,name" in str(e), str(e)[-120:])
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
