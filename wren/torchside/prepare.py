"""Stage prepare (torch environment): audio folders -> decoded 32 kHz audio and labelled 3 s windows, per split.
Usage: python -m wren.torchside.prepare SPEC.json OUT_DIR

SPEC.json (written by wren/stages/data.py): classes (folder names, label order, negative last), negative (index or -1),
sources ({"root": dir} or {"train": dir, "validation": dir|null, "test": dir|null}), split {validation, test, seed},
group (file|subfolder), windows {hop_seconds, max_per_file, negative_max_per_file}. A split ratio of exactly 0 carves
nothing for that split and allows it to stay empty (every file in train: the bird accuracy measurement,
specs/006-bird-accuracy-remeasure); ratios above 0 carve at least one group.

Outputs in OUT_DIR, per split (train, validation, test):
  audio_<split>.raw     every file once: int16, 32 kHz, mono, 48 Hz high-pass as on the device; short files extended
  files_<split>.json    per file: path, class, label, offset/length in audio_<split>.raw, sample rate, bandwidth, flags
  windows_<split>.json  [file index, start sample, label]: the labelled 3 s windows (views of audio_<split>.raw)
  prepare.json          counts, durations, warnings; problems that would make training meaningless stop the stage
"""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio

SR, WIN = 32000, 96000
EXT = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}
OTHER_AUDIO = {".m4a", ".aac", ".opus", ".wma", ".mp4", ".amr"}  # not readable here: reported, not silently skipped
MAX_SECONDS = 30 * 60      # longer files: split them (memory)
PROMINENCE_DB = 6.0        # an event window must stand this far above the file's noise
spec, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
classes, negative = spec["classes"], spec["negative"]
errors, warnings = [], []


def audio_files(folder):
    """Audio files under folder, any case of extension (AudioMoth writes .WAV), sorted: independent of directory order."""
    every = [p for p in Path(folder).rglob("*") if p.is_file() and not p.name.startswith(".")]
    other = [p for p in every if p.suffix.lower() in OTHER_AUDIO]
    if other: warnings.append(f"{len(other)} files in {folder} have a format that cannot be read ({other[0].suffix}); convert them to .wav or .flac")
    return sorted(p for p in every if p.suffix.lower() in EXT)


def group_of(path, class_dir):
    if spec["group"] == "subfolder":
        rel = path.relative_to(class_dir)
        return str(class_dir / rel.parts[0]) if len(rel.parts) > 1 else str(path)
    return str(path)


# --- 1. files per split and class ---------------------------------------------------------------------------------
src = spec["sources"]
by_split = {s: {c: [] for c in classes} for s in ("train", "validation", "test")}
EMPTY_OK = {s for s in ("validation", "test") if spec["split"][s] == 0 and not src.get(s)}  # ratio 0: none carved


def carve(ratio, groups):
    return 0 if ratio == 0 else max(1, round(ratio * groups))


if "root" in src:
    rnd = random.Random(spec["split"]["seed"])
    for c in classes:
        cdir = Path(src["root"]) / c
        groups = defaultdict(list)
        for f in audio_files(cdir): groups[group_of(f, cdir)].append(f)
        keys = sorted(groups)
        rnd.shuffle(keys)
        g = len(keys)
        if g < 5 and len(EMPTY_OK) < 2:
            errors.append(f"class '{c}': {g} {'files' if spec['group'] == 'file' else 'groups'}, at least 5 are needed"); continue
        n_test = carve(spec["split"]["test"], g); n_val = carve(spec["split"]["validation"], g)
        for i, k in enumerate(keys):
            s = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
            by_split[s][c] += groups[k]
else:
    given = {s: src.get(s) for s in ("train", "validation", "test")}
    carve_test = not given["test"] and not given["validation"]  # only train: test and validation are carved from it
    if not given["test"] and given["validation"]:
        # the given validation becomes the test; validation is carved from train, so model selection never sees the test
        given["test"], given["validation"] = given["validation"], None
        warnings.append("no test folder: the given validation folder is used as the test split, validation is carved from train")
    for s, d in given.items():
        if not d: continue
        have = sorted(p.name for p in Path(d).iterdir() if p.is_dir() and not p.name.startswith("."))
        if set(have) != set(classes):
            errors.append(f"{s} folder {d}: classes {have} differ from {classes}")
        for c in classes:
            if (Path(d) / c).is_dir(): by_split[s][c] = audio_files(Path(d) / c)
    rnd = random.Random(spec["split"]["seed"])
    for c in classes:
        if given["validation"]: continue
        cdir = Path(given["train"]) / c
        groups = defaultdict(list)
        for f in by_split["train"][c]: groups[group_of(f, cdir)].append(f)
        keys = sorted(groups); rnd.shuffle(keys)
        n_val = carve(spec["split"]["validation"], len(keys))
        n_test = carve(spec["split"]["test"], len(keys)) if carve_test else 0
        by_split["test"][c] = by_split["test"][c] if not carve_test else [f for k in keys[:n_test] for f in groups[k]]
        by_split["validation"][c] = [f for k in keys[n_test:n_test + n_val] for f in groups[k]]
        by_split["train"][c] = [f for k in keys[n_test + n_val:] for f in groups[k]]
for s in by_split:
    for c in classes:
        if not by_split[s][c] and s not in EMPTY_OK: errors.append(f"class '{c}' has no files in the {s} split")
if errors: sys.exit("prepare: " + "\n  ".join(errors))

# --- 2. exact duplicates: across splits or classes they would inflate the test score ------------------------------
seen = {}
for s in by_split:
    for c in classes:
        keep = []
        for f in by_split[s][c]:
            h = hashlib.sha1(f.read_bytes()).hexdigest()
            if h in seen:
                s0, c0, f0 = seen[h]
                if (s0, c0) != (s, c): errors.append(f"duplicate file in {s0}/{c0} and {s}/{c}: {f0} = {f}")
                else: warnings.append(f"duplicate file ignored: {f} = {f0}")
                continue
            seen[h] = (s, c, f); keep.append(f)
        by_split[s][c] = keep
if errors: sys.exit("prepare: " + "\n  ".join(errors[:20]))


# --- 3. decode, resample, high-pass, store; choose windows ---------------------------------------------------------
def decode(path):
    """Mono float32 at the file's rate, read in blocks. (libsndfile: wav, flac, ogg, mp3, aiff.)"""
    with soundfile.SoundFile(str(path)) as f:
        if f.frames > MAX_SECONDS * f.samplerate:
            raise ValueError(f"{f.frames / f.samplerate / 60:.0f} min long: split files longer than {MAX_SECONDS // 60} min")
        sr, parts = f.samplerate, []
        for block in f.blocks(blocksize=1 << 20, dtype="float32", always_2d=True):
            parts.append(block.mean(axis=1))
    return np.concatenate(parts) if parts else np.zeros(0, np.float32), sr


def bandwidth(x, sr):
    """Upper frequency (Hz) of the RECORDING, at its own sample rate (so resampling images do not count), from up to
    30 s: the highest band whose background level (median over time, so calls do not count) is within 60 dB of the
    strongest band. At most sr / 2. Low-pass recordings and lossy codecs show up here."""
    seg = x[: 30 * sr]
    if len(seg) < 2048: return float("nan")
    frames = torch.from_numpy(np.ascontiguousarray(seg[: len(seg) // 1024 * 1024].reshape(-1, 1024)))
    bg = (torch.fft.rfft(frames * torch.hann_window(1024), dim=1).abs() ** 2).median(dim=0).values[4:]
    alive = torch.nonzero(bg > bg.max() * 1e-6).flatten()
    return min(sr / 2, float((alive.max().item() + 4) * sr / 1024)) if len(alive) else float("nan")


LO_BIN, HI_BIN = int(np.ceil(150 * 1024 / SR)), int(16000 * 1024 / SR)  # FFT bins within the model's 150-16000 Hz
BAND_HZ = np.arange(LO_BIN, HI_BIN + 1) * SR / 1024.0


def prominence(x):
    """Per band and 10 ms frame, dB above the band's own floor over the file (so loud noise in one band does not count
    as an event in another). The floor is the 20th percentile, not the median: an animal calling in more than half of
    the frames (a chorus) must not become the floor."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    S = torch.stft(t, n_fft=1024, hop_length=320, window=torch.hann_window(1024), return_complex=True).abs() ** 2
    db = 10 * torch.log10(S[LO_BIN:HI_BIN + 1] + 1e-12)
    floor = db.kthvalue(max(1, int(0.2 * db.shape[1])), dim=1, keepdim=True).values
    return torch.clamp(db - floor, min=0)


def presence(x):
    """Per band, the share of the file's 3 s windows with an event in that band: at least 3 frames 20 dB above the
    band's floor (noise alone gets there with probability ~1e-9 per frame). Per window, so long and short files count
    alike."""
    above = (prominence(x) >= 20.0).numpy()
    grid = range(0, max(1, above.shape[1] - 300), 300)
    return np.mean([above[:, a:a + 301].sum(1) >= 3 for a in grid], axis=0)


def event_scores(x, starts, band=None):
    """Prominence of events in dB, so that loud noise does not outweigh a call: per frame, the mean of the 8 bands that
    stand highest above their median; per window, the mean of its 30 most prominent frames. Only the bands within `band`
    (Hz, the class's band, see 3b) count, so wind or rain outside the animal's range is ignored. Returns the window
    scores and the file's noise reference (the 20th percentile of the frames)."""
    above = prominence(x)
    if band is not None: above = above[torch.from_numpy((BAND_HZ >= band[0]) & (BAND_HZ <= band[1]))]
    frame = above.topk(min(8, above.shape[0]), dim=0).values.mean(0).numpy()
    scores = np.array([float(np.sort(frame[s0 // 320: s0 // 320 + 301])[-30:].mean()) for s0 in starts])
    return scores, float(np.percentile(frame, 20))


hop = int(round(spec["windows"]["hop_seconds"] * SR))
neg_hop = WIN  # background windows on a 3 s grid (independent of the device settings, so changing those does not retrain)
summary = {"classes": classes, "negative": negative, "splits": {}, "warnings": warnings}
present, stored = defaultdict(list), {}
stats = defaultdict(lambda: {"files": 0, "seconds": 0.0, "windows": 0, "sample_rates": Counter(), "bandwidth": [],
                             "padded_files": 0, "no_clear_event": 0, "clipped_files": 0, "unreadable": 0})
for s in ("train", "validation", "test"):
    raw = open(out / f"audio_{s}.raw", "wb")
    files, windows, offset = [], [], 0
    for label, c in enumerate(classes):
        st = stats[(s, c)]
        for path in by_split[s][c]:
            try:
                x, sr = decode(path)
            except Exception as e:
                st["unreadable"] += 1; warnings.append(f"unreadable, skipped: {path} ({e})"); continue
            if len(x) == 0:
                st["unreadable"] += 1; warnings.append(f"empty, skipped: {path}"); continue
            native_bandwidth = bandwidth(x, sr)
            full_scale = int((np.abs(x) >= 0.999).sum())  # clipped in the file (by the recorder or when it was written)
            t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
            if sr != SR:  # high-quality resampling: the default leaves images above the original Nyquist (~-43 dB)
                t = torchaudio.functional.resample(t, sr, SR, resampling_method="sinc_interp_kaiser", lowpass_filter_width=64,
                                                   rolloff=0.9475, beta=14.769656459379492)
            t = torchaudio.functional.highpass_biquad(t, SR, 48.0)  # the device's DC-blocking filter
            x = t[0].numpy()
            clipped = full_scale + int((np.abs(x) > 1.0).sum())  # or by the resampling and filtering here
            if clipped >= 10: st["clipped_files"] += 1
            seconds = len(x) / SR
            padded = len(x) < WIN
            if padded:  # extend by reflection (never digital silence, which the device does not see; no jumps)
                pieces, k = [], 0
                while sum(len(q) for q in pieces) < WIN:
                    pieces.append(x if k % 2 == 0 else x[::-1]); k += 1
                x = np.concatenate(pieces)[:WIN]
                st["padded_files"] += 1
            pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
            raw.write(pcm.tobytes())
            chosen = []
            if label == negative:  # a 3 s grid, evenly thinned to the cap (training also draws other positions)
                grid = list(range(0, len(pcm) - WIN + 1, neg_hop)) or [0]
                cap = spec["windows"]["negative_max_per_file"]
                chosen = [grid[int(i * len(grid) / cap)] for i in range(cap)] if len(grid) > cap else grid
            if s == "train":  # in which bands this file has events: gives the class's band (3b)
                present[label].append(presence(pcm.astype(np.float32) / 32768.0))
            fi = len(files)
            files.append({"path": str(path), "class": c, "label": label, "offset": offset, "length": len(pcm),
                          "seconds": round(seconds, 3), "sample_rate": sr, "bandwidth_hz": round(native_bandwidth, 1),
                          "padded": padded, "clipped_samples": clipped})
            windows += [[fi, int(w), label] for w in chosen]
            offset += len(pcm)
            st["files"] += 1; st["seconds"] += seconds; st["windows"] += len(chosen)
            st["sample_rates"][sr] += 1; st["bandwidth"].append(files[-1]["bandwidth_hz"])
    raw.close()
    stored[s] = (files, windows, offset)

# --- 3b. event windows of the positive classes, scored within each class's band -----------------------------------
# The band is learned from the TRAINING files: per FFT bin, the share of the class's 3 s windows with an event there,
# minus the same share in the negative class (the class meant to hold the site's noise), or, without one, the median
# over the other classes. Noise that the reference has too (wind, rain, hum) cancels out; the animal's band remains.
# The band is +- 1 octave around the highest contrast (per bin: a tonal call may fill only a few bins). The whole range
# is scored instead when the contrast is weak (< 0.05) or ambiguous (bins over 2 octaves away reach 3/4 of the maximum:
# events in several bands, or a broadband call). Noise found only in one class's recordings cannot be told apart from
# the animal: a band below 1 kHz is reported, so that it can be checked.
bands = {}
profile = {k: np.mean(v, axis=0) for k, v in present.items() if v}
for label, c in enumerate(classes):
    if label == negative or label not in profile or len(profile) < 2: continue
    ref = profile[negative] if negative in profile else np.median([v for k, v in profile.items() if k != label], axis=0)
    contrast = profile[label] - ref
    if contrast.max() < 0.05: continue
    mid = float(BAND_HZ[int(np.argmax(contrast))])
    lo, hi = max(150.0, mid / 2), min(16000.0, mid * 2)
    outside = np.abs(np.log2(BAND_HZ / mid)) > 2  # more than 2 octaves away: another band, not the same one's edge
    if outside.any() and contrast[outside].max() >= 0.75 * contrast.max():
        far = float(BAND_HZ[outside][int(np.argmax(contrast[outside]))])
        warnings.append(f"class '{c}': events in more than one band (around {mid:.0f} and {far:.0f} Hz); its windows "
                        "are scored over the whole range")
        continue
    bands[label] = (lo, hi)
    if mid < 1000:
        warnings.append(f"class '{c}': its events lie around {mid:.0f} Hz (windows chosen within {lo:.0f}-{hi:.0f} Hz); "
                        "if the animal does not call there, this is noise found only in its recordings (wind, traffic, "
                        "handling): add background recorded at the same sites")
summary["class_bands_hz"] = {classes[k]: [round(v[0]), round(v[1])] for k, v in bands.items()}
for s in ("train", "validation", "test"):
    files, windows, offset = stored[s]
    audio = np.memmap(out / f"audio_{s}.raw", dtype=np.int16, mode="r", shape=(offset,)) if offset else None
    for fi, f in enumerate(files):
        label = f["label"]
        if label == negative: continue
        pcm = audio[f["offset"]: f["offset"] + f["length"]].astype(np.float32) / 32768.0
        starts = list(range(0, f["length"] - WIN + 1, hop)) or [0]
        sc, noise_ref = event_scores(pcm, starts, bands.get(label))
        order = np.argsort(-sc, kind="stable")
        chosen = []
        for i in order:  # at least PROMINENCE_DB above the file's noise, the strongest first, 3 s apart
            if sc[i] < noise_ref + PROMINENCE_DB or len(chosen) >= spec["windows"]["max_per_file"]: break
            if all(abs(starts[i] - c0) >= WIN for c0 in chosen): chosen.append(starts[i])
        st = stats[(s, f["class"])]
        if not chosen:
            if f["length"] < 2 * WIN: chosen = [starts[order[0]]]  # a clip under 6 s: the clip is the label (kept)
            else: st["no_clear_event"] += 1  # a long file without an event: no windows
        windows += [[fi, int(w), label] for w in sorted(chosen)]
        st["windows"] += len(chosen)
    windows.sort(key=lambda w: (w[0], w[1]))
    (out / f"files_{s}.json").write_text(json.dumps(files))
    (out / f"windows_{s}.json").write_text(json.dumps(windows))
    summary["splits"][s] = {"files": len(files), "windows": len(windows), "samples": offset}

# --- 4. report and checks -----------------------------------------------------------------------------------------
per_class = {}
for (s, c), st in stats.items():
    if s in EMPTY_OK and st["files"] == 0: continue
    bw = [b for b in st["bandwidth"] if b == b]
    per_class.setdefault(c, {})[s] = {"files": st["files"], "minutes": round(st["seconds"] / 60, 2), "windows": st["windows"],
                                      "sample_rates": dict(st["sample_rates"]), "median_bandwidth_hz": round(float(np.median(bw)), 0) if bw else None,
                                      "padded_files": st["padded_files"], "files_without_clear_event": st["no_clear_event"],
                                      "clipped_files": st["clipped_files"], "unreadable": st["unreadable"]}
    if st["files"] == 0: errors.append(f"class '{c}' has no readable files in the {s} split")
    elif st["windows"] == 0: errors.append(f"class '{c}' has no windows with an event in the {s} split")
    tried = st["files"] + st["unreadable"]
    if tried and st["unreadable"] > 0.1 * tried: errors.append(f"class '{c}': {st['unreadable']} of {tried} files unreadable in the {s} split")
    if c != classes[negative] if negative >= 0 else True:
        if st["files"] and st["no_clear_event"] > 0.3 * st["files"]:
            warnings.append(f"class '{c}' ({s}): {st['no_clear_event']} of {st['files']} files (6 s or longer) have no "
                            f"clear event ({PROMINENCE_DB:.0f} dB above the file's noise) and give no windows")
summary["per_class"] = per_class
bws = {c: v["train"]["median_bandwidth_hz"] for c, v in per_class.items() if v.get("train", {}).get("median_bandwidth_hz")}
if bws and max(bws.values()) > 1.25 * min(bws.values()):
    warnings.append(f"classes were recorded with different bandwidths (median upper frequency of the recordings, Hz: {bws}); "
                    "the model may learn the recording instead of the animal")
rates = sorted({r for v in per_class.values() for sp in v.values() for r in sp["sample_rates"]})
if rates and min(rates) < SR:
    warnings.append(f"files at {min(rates)} Hz: nothing above {min(rates) // 2} Hz in them, while the AudioMoth records up to 16 kHz")
clip = {c: sum(sp["clipped_files"] for sp in v.values()) for c, v in per_class.items()}
if any(clip.values()):
    warnings.append(f"files with clipped samples per class: { {c: n for c, n in clip.items() if n} }; clipping spreads "
                    "loud noise (wind, handling) over all frequencies, where it can be chosen as an event")
pads = {c: v["train"]["padded_files"] / max(1, v["train"]["files"]) for c, v in per_class.items()}
if pads and max(pads.values()) - min(pads.values()) > 0.3:
    warnings.append(f"short (padded) files are much more common in some classes: {pads}")
(out / "prepare.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
for w in warnings[:20]: print("warning:", w)
print(json.dumps(summary["splits"]))
if errors: sys.exit("prepare: " + "\n  ".join(errors))
