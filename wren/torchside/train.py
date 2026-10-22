"""Stage train (torch environment): WrenNet on the windows of prepare.
Usage: python -m wren.torchside.train SPEC.json PREPARE_DIR OUT_DIR

SPEC.json: classes, negative, init (wrennet|scratch), wrennet_checkpoint, epochs, batch_size, lr, negative_share, loss
(ce|focal), focal_gamma, augment, seed, device (auto|cpu|mps|cuda), patience, warmup_epochs.
Outputs: best.pt (plain state_dict, loads with wren.torchside.model.load_model), history.csv, train.json.

Choices (README.md, project.yaml → training):
- per-clip front-end (device_forward), so a batch gives what the device computes one window at a time;
- init wrennet: every weight of the bird model but the classifier; the learned filterbank parameters stay frozen
  (the C tables are the bird model's); the first warmup_epochs train the classifier only;
- init scratch: the filterbank parameters train with WrenNet's learning rates (3.0, 0.1), no weight decay, and the
  filterbank cache is rebuilt before every step;
- batches: negative_share of background windows, the other classes equally, files equally within a class;
  cross-entropy (focal optional); classifier bias = log of these shares;
- augmentation: offset jitter within the file, background mix 5-20 dB, speed +-5 %, random low-pass, SpecAugment;
- cosine learning rate; the best epoch by validation macro-F1 (tie: lower validation loss); early stopping.
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from wren.torchside.metrics import scores
from wren.torchside.model import Core, build_model, device_forward, frontend, load_model

WIN, SR = 96000, 32000
spec = json.loads(Path(sys.argv[1]).read_text())
prep, out = Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
classes, neg = spec["classes"], spec["negative"]
K = len(classes)

# --- reproducibility --------------------------------------------------------------------------------------------
seed = int(spec["seed"])
torch.manual_seed(seed)
rng = np.random.default_rng(seed)
def usable(name):
    """A GPU that is reported but cannot allocate (virtual machines, e.g. CI's macOS runners) is not usable."""
    try:
        torch.ones(1024, device=name).sum().item()
        return True
    except RuntimeError as e:
        print(f"note: {name} is reported but does not work ({str(e).splitlines()[0]}); training on the cpu")
        return False


dev = spec["device"]
if dev == "auto":
    dev = next((d for d, ok in (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available()))
                if ok and usable(d)), "cpu")
device = torch.device(dev)
if dev == "cpu":
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


# --- data ---------------------------------------------------------------------------------------------------------
class Split:
    def __init__(self, name):
        self.files = json.loads((prep / f"files_{name}.json").read_text())
        self.windows = json.loads((prep / f"windows_{name}.json").read_text())
        n = sum(f["length"] for f in self.files)
        self.audio = np.memmap(prep / f"audio_{name}.raw", dtype=np.int16, mode="r", shape=(n,))

    def window(self, fi, start):
        f = self.files[fi]
        return self.audio[f["offset"] + start: f["offset"] + start + WIN].astype(np.float32) / 32768.0


train, val = Split("train"), Split("validation")
by_class = [{} for _ in range(K)]  # class -> file -> windows
for fi, start, label in train.windows: by_class[label].setdefault(fi, []).append(start)
for c in range(K):
    if not by_class[c]: sys.exit(f"train: no training windows for class '{classes[c]}'")
share = np.full(K, 1.0 / K)
if neg >= 0 and K > 1:
    share = np.full(K, (1.0 - spec["negative_share"]) / (K - 1)); share[neg] = spec["negative_share"]
neg_windows = [(fi, s) for fi, ws in by_class[neg].items() for s in ws] if neg >= 0 else []
_draws = max(20, math.ceil(len(train.windows) / spec["batch_size"])) * spec["batch_size"]
for c in range(K):
    n_c = sum(len(w) for w in by_class[c].values())
    if c != neg and share[c] * _draws > 10 * n_c:
        print(f"warning: class '{classes[c]}' has {n_c} training windows, each drawn ~{share[c] * _draws / n_c:.0f} times per epoch "
              "(more recordings would help)", flush=True)


def draw(n):
    """n training windows: class by share, then a file of that class uniformly, then one of its windows; background
    windows at any position of their file, so all of the background audio is used over the epochs."""
    labels = rng.choice(K, size=n, p=share)
    picks = []
    for c in labels:
        files = list(by_class[c])
        fi = files[rng.integers(len(files))]
        if c == neg:
            start = int(rng.integers(0, max(1, train.files[fi]["length"] - WIN + 1)))
        else:
            start = by_class[c][fi][rng.integers(len(by_class[c][fi]))]
        picks.append((fi, start, int(c)))
    return picks


def augmented(fi, start):
    f = train.files[fi]
    if spec["augment"]:
        jitter = int(rng.integers(-SR // 2, SR // 2 + 1))  # the device's windows are not centred on calls
        speed = float(rng.uniform(0.95, 1.05))
        need = int(round(WIN * speed))
        start = int(np.clip(start + jitter, 0, max(0, f["length"] - need)))
        seg = train.audio[f["offset"] + start: f["offset"] + start + need].astype(np.float32) / 32768.0
        if len(seg) < need: seg = np.pad(seg, (0, need - len(seg)), mode="reflect")
        x = torch.from_numpy(seg)
        if need != WIN: x = F.interpolate(x[None, None], size=WIN, mode="linear", align_corners=False)[0, 0]
        if neg_windows and fi not in by_class[neg] and rng.random() < 0.5:
            bfi, bs = neg_windows[rng.integers(len(neg_windows))]
            bg = torch.from_numpy(train.window(bfi, bs))
            snr = float(rng.uniform(5, 20))
            ps, pb = x.pow(2).mean() + 1e-12, bg.pow(2).mean() + 1e-12
            x = torch.clamp(x + bg * torch.sqrt(ps / (pb * 10 ** (snr / 10))), -1.0, 1.0)
        if rng.random() < 0.3:
            x = torchaudio.functional.lowpass_biquad(x, SR, float(rng.uniform(6000, 14000)))
        return x
    return torch.from_numpy(train.window(fi, start))


def spec_augment(s):
    """Time and frequency masks on the standardised spectrogram [B, 64, T] (0 = its mean)."""
    s = s.clone()
    for b in range(s.shape[0]):
        if rng.random() < 0.5:
            for _ in range(2):
                w = int(rng.integers(1, 21)); t0 = int(rng.integers(0, s.shape[2] - w)); s[b, :, t0:t0 + w] = 0
        if rng.random() < 0.5:
            for _ in range(2):
                w = int(rng.integers(1, 9)); f0 = int(rng.integers(0, s.shape[1] - w)); s[b, f0:f0 + w, :] = 0
    return s


# --- model --------------------------------------------------------------------------------------------------------
m = build_model(K)
fl = m.combined_log_linear_spec
if spec["init"] == "wrennet":
    sd = torch.load(spec["wrennet_checkpoint"], map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        sys.exit(f"train: the WrenNet checkpoint does not fit (missing {missing}, unexpected {unexpected})")
    fl.breakpoint.requires_grad_(False); fl.transition_width.requires_grad_(False)
elif spec["init"] != "scratch":
    sys.exit(f"train: init must be wrennet or scratch, not {spec['init']}")
with torch.no_grad():
    m.fc.bias.copy_(torch.log(torch.tensor(share, dtype=torch.float32)))
m.to(device)
core = Core(m)
frontend_params = [fl.breakpoint, fl.transition_width]
main_params = [p for n, p in m.named_parameters() if p.requires_grad and n not in
               ("combined_log_linear_spec.breakpoint", "combined_log_linear_spec.transition_width")]
groups = [{"params": main_params, "lr": spec["lr"], "weight_decay": 1e-4}]
if spec["init"] == "scratch":
    groups += [{"params": [fl.breakpoint], "lr": 3.0, "weight_decay": 0.0},
               {"params": [fl.transition_width], "lr": 0.1, "weight_decay": 0.0}]
opt = torch.optim.AdamW(groups)
steps_per_epoch = max(20, math.ceil(len(train.windows) / spec["batch_size"]))
total = steps_per_epoch * spec["epochs"]
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, total) / total)))
gamma = float(spec.get("focal_gamma", 2.0)) if spec["loss"] == "focal" else 0.0


def loss_fn(logits, y):
    ce = F.cross_entropy(logits, y, reduction="none")
    return (ce if gamma == 0 else (1 - torch.exp(-ce)) ** gamma * ce).mean()


def evaluate_split(split):
    m.eval()
    ys, ps, losses = [], [], []
    with torch.no_grad():
        for i in range(0, len(split.windows), 64):
            chunk = split.windows[i:i + 64]
            x = torch.from_numpy(np.stack([split.window(fi, s) for fi, s, _ in chunk])).to(device)
            y = torch.tensor([l for _, _, l in chunk], device=device)
            logits = device_forward(m, core, x)
            losses.append(float(F.cross_entropy(logits, y, reduction="sum")))
            ys += y.tolist(); ps += logits.argmax(1).tolist()
    r = scores(ys, ps, K, classes)
    r["loss"] = sum(losses) / max(1, len(ys))
    return r


# --- loop ---------------------------------------------------------------------------------------------------------
history, best, bad, t0 = [], None, 0, time.time()
warmup = int(spec.get("warmup_epochs", 2)) if spec["init"] == "wrennet" else 0
for epoch in range(1, spec["epochs"] + 1):
    m.train()
    for n, p in m.named_parameters():  # warm-up: only the new classifier learns
        if n in ("combined_log_linear_spec.breakpoint", "combined_log_linear_spec.transition_width"):
            continue
        p.requires_grad_(epoch > warmup or n.startswith("fc."))
    tl, tn = 0.0, 0
    for _ in range(steps_per_epoch):
        batch = draw(spec["batch_size"])
        x = torch.stack([augmented(fi, s) for fi, s, _ in batch]).to(device)
        y = torch.tensor([c for _, _, c in batch], device=device)
        if spec["init"] == "scratch": fl.rebuild_filters()  # the cache would drop the filterbank's gradient
        s, _ = frontend(m, x)
        if spec["augment"]: s = spec_augment(s)
        loss = loss_fn(core(s), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tl += float(loss.detach()) * len(batch); tn += len(batch)
    if spec["init"] == "scratch": fl.rebuild_filters()
    v = evaluate_split(val)
    row = {"epoch": epoch, "train_loss": round(tl / tn, 5), "val_loss": round(v["loss"], 5), "val_accuracy": round(v["accuracy"], 4),
           "val_macro_f1": round(v["macro_f1"], 4), "breakpoint": round(float(fl.breakpoint), 3),
           "transition_width": round(float(fl.transition_width), 3), "seconds": round(time.time() - t0, 1)}
    history.append(row)
    print(json.dumps(row), flush=True)
    better = best is None or (v["macro_f1"], -v["loss"]) > best_raw  # raw values: the rounded ones in `row` would break ties
    if better:
        best, best_raw, bad = dict(row), (v["macro_f1"], -v["loss"]), 0
        torch.save({k: t.detach().cpu() for k, t in m.state_dict().items()}, out / "best.pt")
    else:
        bad += 1
        if bad >= spec["patience"]: print(f"early stop at epoch {epoch}"); break

with open(out / "history.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
# the saved model must load like any checkpoint and have usable front-end parameters for the C tables
check = load_model(out / "best.pt")
bp, tw = float(check.combined_log_linear_spec.breakpoint.detach()), float(check.combined_log_linear_spec.transition_width.detach())
if not (math.isfinite(bp) and math.isfinite(tw) and tw > 0 and 150 < bp < 16000):
    sys.exit(f"train: learned front-end parameters out of range (breakpoint {bp}, transition width {tw})")
info = {"best_epoch": best["epoch"], "best": best, "epochs_run": len(history), "device": dev, "init": spec["init"],
        "seed": seed, "classes": classes, "negative": neg, "train_windows": len(train.windows), "validation_windows": len(val.windows),
        "seconds": round(time.time() - t0, 1)}
(out / "train.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info))
