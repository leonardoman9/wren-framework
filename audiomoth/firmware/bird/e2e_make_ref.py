"""PyTorch reference for the whole model on int16 audio. Usage: python e2e_make_ref.py CKPT OUT.npz N DIR_OR_FILE..."""
import random, sys
from pathlib import Path
import numpy as np, soundfile, torch, torchaudio
ckpt, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repository: wren's model loader
from wren.torchside.model import SR, load_model  # noqa: E402
# float64, rounded once to float32: MKL's last bits differ between CPUs, and they vanish in the rounding
m = load_model(ckpt).double()
files = []
for a in sys.argv[4:]:
    p = Path(a); files += (sorted(p.rglob("*.wav")) + sorted(p.rglob("*.mp3")) + sorted(p.rglob("*.flac"))) if p.is_dir() else [p]
random.Random(7).shuffle(files)
pcms, logits, names = [], [], []
for f in files:
    if len(pcms) >= n: break
    try: d, sr = soundfile.read(str(f), dtype="float32", always_2d=True)
    except Exception: continue
    w = torchaudio.functional.resample(torch.from_numpy(d.T.copy()).mean(0, keepdim=True), sr, SR)
    # up to 3 non-overlapping 3 s windows per file
    for s in range(0, max(1, w.shape[1] - 96000 + 1), 96000)[:3]:
        if len(pcms) >= n: break
        seg = torch.nn.functional.pad(w[:, s:s + 96000], (0, max(0, 96000 - w[:, s:s + 96000].shape[1])))
        pcm = np.clip(np.rint(seg[0].numpy() * 32768.0), -32768, 32767).astype(np.int16)
        with torch.no_grad(): lg = m(torch.from_numpy(pcm.astype(np.float64) / 32768.0)[None])[0].numpy().astype(np.float32)
        pcms.append(pcm); logits.append(lg); names.append(f"{f.name}@{s/SR:.0f}s")
np.savez(out, pcm=np.stack(pcms), logits=np.stack(logits), names=np.array(names))
print("saved", len(pcms), "windows")
