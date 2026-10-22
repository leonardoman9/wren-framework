"""PyTorch reference on labelled windows of a wren-trained model (torch environment), in the format of the firmware's
e2e_make_ref.py (pcm int16, logits, names) plus labels.
Usage: python -m wren.torchside.windows_ref PREPARE_DIR SPLIT CHECKPOINT.pt OUT.npz MAX_WINDOWS SEED"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from wren.torchside.model import Core, device_forward, load_model

WIN, SR = 96000, 32000
prep, split, ckpt, out, n, seed = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
# float64, rounded once to float32: MKL's last bits differ between CPUs, and they vanish in the rounding
m = load_model(ckpt).double()
core = Core(m).eval()
files = json.loads((prep / f"files_{split}.json").read_text())
windows = json.loads((prep / f"windows_{split}.json").read_text())
audio = np.memmap(prep / f"audio_{split}.raw", dtype=np.int16, mode="r", shape=(sum(f["length"] for f in files),))
random.Random(seed).shuffle(windows)
windows = windows[:n]
pcm = np.stack([np.array(audio[files[fi]["offset"] + s: files[fi]["offset"] + s + WIN]) for fi, s, _ in windows])
logits = []
with torch.no_grad():
    for i in range(0, len(pcm), 64):
        logits.append(device_forward(m, core, torch.from_numpy(pcm[i:i + 64].astype(np.float64) / 32768.0)).numpy().astype(np.float32))
np.savez(out, pcm=pcm, logits=np.concatenate(logits), labels=np.array([l for _, _, l in windows]),
         names=np.array([f"{Path(files[fi]['path']).name}@{s / SR:.1f}s" for fi, s, _ in windows]))
print("saved", len(pcm), "windows")
