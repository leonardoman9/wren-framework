"""Calibration and evaluation spectrograms for the int8 conversion of a model trained by wren (torch environment).
Usage: python -m wren.torchside.specs_from_windows PREPARE_DIR CHECKPOINT.pt OUT.npz N_CALIB N_EVAL SEED DEVICE_HOP_S

calib: windows on the device grid of the TRAINING files (what the device will see: calls, noise, everything between),
       taken round-robin over the classes so every class contributes; eval: the validation call windows.
Same format as make_specs.py: calib/eval [N, 301, 64] float32 standardised spectrograms, names."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from wren.torchside.model import frontend, load_model

WIN, SR = 96000, 32000
prep, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
n_cal, n_eval, seed, hop = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(round(float(sys.argv[7]) * SR))
# float64, rounded once to float32: MKL's last bits differ between CPUs, and they vanish in the rounding
m = load_model(ckpt).double()


def split(name):
    files = json.loads((prep / f"files_{name}.json").read_text())
    audio = np.memmap(prep / f"audio_{name}.raw", dtype=np.int16, mode="r", shape=(sum(f["length"] for f in files),))
    return files, json.loads((prep / f"windows_{name}.json").read_text()), audio


def specs(items, files, audio):
    res = []
    with torch.no_grad():
        for i in range(0, len(items), 64):
            x = np.stack([audio[files[fi]["offset"] + s: files[fi]["offset"] + s + WIN].astype(np.float64) / 32768.0 for fi, s in items[i:i + 64]])
            res.append(frontend(m, torch.from_numpy(x))[0].transpose(1, 2).numpy().astype(np.float32))
    return np.concatenate(res) if res else np.zeros((0, 301, 64), np.float32)


files, _, audio = split("train")
rnd = random.Random(seed)
per_class = {}
for fi, f in enumerate(files):
    per_class.setdefault(f["label"], []).extend((fi, s) for s in (range(0, f["length"] - WIN + 1, hop) or [0]))
for v in per_class.values(): rnd.shuffle(v)
order = [per_class[c] for c in sorted(per_class)]
calib = []
while len(calib) < n_cal and any(order):
    for lst in order:
        if lst and len(calib) < n_cal: calib.append(lst.pop())
vfiles, vwin, vaudio = split("validation")
ev = [(fi, s) for fi, s, _ in vwin]
rnd.shuffle(ev)
ev = ev[:n_eval]
np.savez(out, calib=specs(calib, files, audio), eval=specs(ev, vfiles, vaudio),
         names=np.array([f"{files[fi]['path']}@{s / SR:.1f}s" for fi, s in calib]))
print(f"calib {len(calib)} windows from {len(per_class)} classes, eval {len(ev)} validation windows")
