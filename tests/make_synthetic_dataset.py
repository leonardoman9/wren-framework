"""A small synthetic dataset for the training smoke test: three tonal "species" and a background class.
Usage: python tests/make_synthetic_dataset.py OUT_DIR [FILES_PER_CLASS]      (only numpy and soundfile)

  upsweep/    calls rising 2 -> 5 kHz        downsweep/  calls falling 7 -> 3 kHz
  trill/      fast 9 kHz pulse trains        background/ coloured noise only
Every file is 4-8 s at 32 kHz with 1-4 calls at random times over noise at a random level, so the classes are easy
but not trivial: a working pipeline must learn them almost perfectly; a broken one cannot."""
import sys
from pathlib import Path

import numpy as np
import soundfile

out, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 24
fs = 32000
rng = np.random.default_rng(1)


def noise(length):
    x = np.cumsum(rng.normal(0, 1, length))
    x -= np.convolve(x, np.ones(64) / 64, "same")  # crude coloured noise, no DC
    return x / (np.std(x) + 1e-9)


def call(kind):
    if kind == "upsweep":
        t = np.arange(int(0.35 * fs)) / fs
        f0, f1 = rng.uniform(1800, 2300), rng.uniform(4600, 5400)
        y = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * t[-1])))
    elif kind == "downsweep":
        t = np.arange(int(0.25 * fs)) / fs
        f0, f1 = rng.uniform(6600, 7400), rng.uniform(2800, 3300)
        y = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * t[-1])))
    else:  # trill: 12 short pulses
        t = np.arange(int(0.6 * fs)) / fs
        f = rng.uniform(8500, 9500)
        y = np.sin(2 * np.pi * f * t) * (np.sin(2 * np.pi * 20 * t) > 0.3)
    return y * np.hanning(len(y))


for kind in ["upsweep", "downsweep", "trill", "background"]:
    (out / kind).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        length = int(rng.uniform(4, 8) * fs)
        x = noise(length) * rng.uniform(0.01, 0.05)
        if kind != "background":
            for _ in range(rng.integers(1, 5)):
                c = call(kind) * rng.uniform(0.1, 0.4)
                s = rng.integers(0, length - len(c))
                x[s:s + len(c)] += c
        soundfile.write(str(out / kind / f"{kind}_{i:03d}.wav"), np.clip(x, -1, 1).astype(np.float32), fs, subtype="PCM_16")
print(f"wrote {4 * n} files to {out}")
