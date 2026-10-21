# Copied unchanged from WrenNet tools/bird_make_specs.py (not published in the WrenNet repository). Run through
# wren/torchside/run.py, which gives it the general model loader (any number of classes).
"""Standardised dB spectrograms ([301, 64], time-major) from audio files, for calibration
and evaluation of the int8 backbone. Takes the first 3 s of every file.
Usage: python tools/bird_make_specs.py CHECKPOINT.pt OUT.npz N_CALIB N_EVAL DIR_OR_FILE [DIR_OR_FILE ...]
Files are shuffled with a fixed seed; the first N_CALIB go to 'calib', the next N_EVAL to 'eval'.
"""
import random
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bird_model_export_check import SR, frontend, load_model  # noqa: E402

ckpt, out, n_cal, n_eval = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
files = []
for arg in sys.argv[5:]:
    p = Path(arg)
    files += sorted(p.rglob("*.wav")) + sorted(p.rglob("*.mp3")) if p.is_dir() else [p]
random.Random(0).shuffle(files)
m = load_model(ckpt)
specs, names = [], []
for path in files:
    if len(specs) >= n_cal + n_eval: break
    try:
        data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        continue
    wav = torchaudio.functional.resample(torch.from_numpy(data.T.copy()).mean(0, keepdim=True), sr, SR)
    wav = torch.nn.functional.pad(wav, (0, max(0, 96000 - wav.shape[1])))[:, :96000]
    with torch.no_grad():
        spec, _ = frontend(m, wav)
    specs.append(spec[0].T.numpy().astype(np.float32))  # [301, 64]
    names.append(str(path))
specs = np.stack(specs)
np.savez(out, calib=specs[:n_cal], eval=specs[n_cal:], names=np.array(names))
print("calib", specs[:n_cal].shape, "eval", specs[n_cal:].shape, "value range", float(specs.min()), float(specs.max()))
