"""Builds the bird model's test bundle, examples/birds/test/ (maintainer tool, run once;
specs/006-bird-accuracy-remeasure/contracts/test-bundle.md). The committed bundle is the reference afterwards: this tool
does not need to reproduce it byte for byte.

Species audio: every freely licensed (CC0, CC BY, CC BY-SA) xeno-canto recording of the model's 8 species, without the
stretches the calibration bundle took from it (each calibration clip +-1.5 s), at most --max-seconds per recording, as
stretches of at least 3 s. The measurement finds the call windows in them itself (prepare's call-finding).
Non-bird audio: CC0 Freesound field recordings (wind, rain, water, insects, traffic, ...), HQ previews, the first 30 s,
kept only if BirdNET hears no species in any 3 s window (above --screen 0.1).

Usage: . ~/.wren-keys && .venv-torch/bin/python tools/make_test_bundle.py OUT_DIR [--max-seconds 120] [--non-bird 40]
       (needs XC_API_KEY, FREESOUND_API_KEY and BirdNET: python -m wren setup --birdnet)"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_calibration_bundle import licence, query, usable  # noqa: E402  (same xeno-canto rules as the calibration)

socket.setdefaulttimeout(60)  # a stalled download fails after 60 s instead of hanging (and is retried)
ROOT = Path(__file__).resolve().parents[1]
SR, MIN_STRETCH, MARGIN = 32000, 3.0, 1.5
FIELDS = ["clip", "source", "source_id", "species", "recordist", "licence", "licence_url", "source_url", "start_s", "end_s",
          "uploaded"]
FREESOUND = "https://freesound.org/apiv2/search/text/"
QUERIES = ["wind", "rain", "stream", "river", "insects", "traffic", "footsteps", "forest ambience"]
CC0_URL = "http://creativecommons.org/publicdomain/zero/1.0/"


def fetch(url, dest, tries=3):
    for i in range(tries):
        try:
            urllib.request.urlretrieve(url, dest); return
        except OSError:
            if i == tries - 1: raise
            time.sleep(5 * (i + 1))


def decode(path):
    """Mono float32 at 32 kHz."""
    data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    return torchaudio.functional.resample(torch.from_numpy(data.T.copy()).mean(0), sr, SR).numpy()


def to_int16(x):
    return np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)


def calibration_spans():
    """xeno-canto id -> the seconds the calibration bundle took from that recording (clip start .. +3 s)."""
    spans = {}
    with (ROOT / "examples/birds/calibration/ATTRIBUTION.csv").open() as f:
        for r in csv.DictReader(f):
            s = float(r["start_s"]); spans.setdefault(int(r["xc_id"]), []).append((s, s + 3.0))
    return spans


def stretches(seconds, taken, max_seconds):
    """The parts of [0, seconds) outside the taken spans (each widened by MARGIN), at least MIN_STRETCH long, in
    order, until max_seconds in total."""
    cuts = sorted((max(0.0, a - MARGIN), min(seconds, b + MARGIN)) for a, b in taken)
    free, at = [], 0.0
    for a, b in cuts + [(seconds, seconds)]:
        if a - at >= MIN_STRETCH: free.append((at, a))
        at = max(at, b)
    out, total = [], 0.0
    for a, b in free:
        if total >= max_seconds: break
        b = min(b, a + max_seconds - total)
        if b - a >= MIN_STRETCH: out.append((a, b)); total += b - a
    return out


def species_audio(out, key, max_seconds):
    """Resumable: every finished recording is appended to OUT/.species.csv (its rows) or .species_done (no stretch)."""
    model = [c for c in yaml.safe_load((ROOT / "examples/birds/project.yaml").read_text())["classes"] if " " in c]
    taken = calibration_spans()
    partial, done_file = out / ".species.csv", out / ".species_done"
    rows = list(csv.DictReader(partial.open())) if partial.exists() else []
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    with tempfile.TemporaryDirectory() as tmp:
        for sp in model:
            recs = sorted(usable(query(sp, key)), key=lambda r: int(r["id"]))
            print(f"{sp}: {len(recs)} usable recordings", flush=True)
            for r in recs:
                if str(r["id"]) in done: continue
                src = Path(tmp) / f"XC{r['id']}"
                try:
                    fetch(r["file"], src)
                    w = decode(src)
                except Exception as e:  # unreadable or unreachable: the measurement does without it
                    print(f"  skip XC{r['id']}: {e}"); continue
                finally:
                    src.unlink(missing_ok=True); time.sleep(0.5)
                name, url = licence(r["lic"])
                folder = out / "audio" / sp.replace(" ", "_") / f"XC{r['id']}"
                new = []
                for k, (a, b) in enumerate(stretches(len(w) / SR, taken.get(int(r["id"]), []), max_seconds)):
                    folder.mkdir(parents=True, exist_ok=True)
                    clip = folder / f"XC{r['id']}_{k}.flac"
                    soundfile.write(str(clip), to_int16(w[int(a * SR): int(b * SR)]), SR, subtype="PCM_16", format="FLAC")
                    new.append({"clip": str(clip.relative_to(out / "audio")), "source": "xeno-canto", "source_id": r["id"],
                                "species": sp, "recordist": r["rec"], "licence": name, "licence_url": url,
                                "source_url": f"https://xeno-canto.org/{r['id']}", "start_s": f"{a:.1f}", "end_s": f"{b:.1f}",
                                "uploaded": r.get("uploaded", "")})
                with partial.open("a", newline="") as f:
                    w_ = csv.DictWriter(f, fieldnames=FIELDS)
                    if f.tell() == 0: w_.writeheader()
                    w_.writerows(new)
                rows += new
                with done_file.open("a") as f: f.write(f"{r['id']}\n")
    return rows


def freesound(query_text, key):
    q = urllib.parse.urlencode({"query": f"{query_text} -bird -birds -birdsong", "page_size": 50, "sort": "rating_desc",
                                "filter": 'license:"Creative Commons 0" duration:[10 TO 600]',
                                "fields": "id,name,username,license,previews,duration,created,url", "token": key})
    return json.load(urllib.request.urlopen(f"{FREESOUND}?{q}", timeout=60))["results"]


def non_bird_audio(out, key, wanted, screen):
    per_query = math.ceil(wanted / len(QUERIES))
    rows, seen = [], set()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        todo = []  # the first 3 x per_query results of each query, in order, each sound once
        for qt in QUERIES:
            n = 0
            for s in freesound(qt, key):
                if s["id"] in seen or n >= 3 * per_query: continue
                seen.add(s["id"]); todo.append((qt, s)); n += 1

        def get(item):  # 30 s of a candidate, kept at 32 kHz and stored at 48 kHz for BirdNET
            qt, s = item
            src = tmp / f"FS{s['id']}.ogg"
            try:
                fetch(s["previews"]["preview-hq-ogg"], src)
                w = decode(src)[: 30 * SR]
            except Exception as e:
                print(f"  skip FS{s['id']}: {e}", flush=True); return None
            finally:
                src.unlink(missing_ok=True)
            if len(w) < 10 * SR: return None
            w48 = torchaudio.functional.resample(torch.from_numpy(w), SR, 48000).numpy()
            np.save(tmp / f"FS{s['id']}.npy", to_int16(w48))
            return s["id"], (qt, s, w)

        with ThreadPoolExecutor(8) as pool:  # downloads wait on the network: 8 at a time
            cand = dict(r for r in pool.map(get, todo) if r)
        for qt in QUERIES:
            print(f"{qt}: {sum(1 for q, _, _ in cand.values() if q == qt)} candidates", flush=True)
        report = tmp / "screen.json"
        r = subprocess.run([str(ROOT / ".venv-tf/bin/python"), str(ROOT / "wren/tfside/birdnet_screen.py"),
                            str(ROOT / "third_party/birdnet/v2.4"), str(tmp), str(report)], capture_output=True, text=True)
        if r.returncode: sys.exit("BirdNET screening failed (python -m wren setup --birdnet?):\n" + r.stderr[-2000:])
        verdict = json.loads(report.read_text())
        kept = {qt: 0 for qt in QUERIES}
        for sid, (qt, s, w) in sorted(cand.items()):
            v = verdict[f"FS{sid}"]
            if v["max_species_confidence"] > screen:
                print(f"  drop FS{sid} ({qt}): BirdNET hears {v['species']} at {v['max_species_confidence']:.2f}"); continue
            if kept[qt] >= per_query or sum(kept.values()) >= wanted: continue
            folder = out / "audio" / "non-bird" / f"FS{sid}"
            folder.mkdir(parents=True, exist_ok=True)
            clip = folder / f"FS{sid}.flac"
            soundfile.write(str(clip), to_int16(w), SR, subtype="PCM_16", format="FLAC")
            kept[qt] += 1
            rows.append({"clip": str(clip.relative_to(out / "audio")), "source": "freesound", "source_id": sid,
                         "species": "non-bird", "recordist": s["username"], "licence": "CC0 1.0", "licence_url": CC0_URL,
                         "source_url": s["url"], "start_s": "0.0", "end_s": f"{len(w) / SR:.1f}",
                         "uploaded": s["created"][:10]})
        print("non-bird kept per query:", kept, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-seconds", type=float, default=120.0)
    ap.add_argument("--non-bird", type=int, default=40)
    ap.add_argument("--screen", type=float, default=0.1)
    a = ap.parse_args()
    if "biodcase" in str(a.out.resolve()).lower(): sys.exit("refusing: BioDCASE must not be used anywhere")
    xc = os.environ.get("XC_API_KEY") or sys.exit("set XC_API_KEY (xeno-canto API key)")
    fs = os.environ.get("FREESOUND_API_KEY") or sys.exit("set FREESOUND_API_KEY (Freesound API key)")
    if (a.out / "audio").exists() and not (a.out / ".species_done").exists():
        sys.exit(f"{a.out / 'audio'} exists but holds no resumable run: remove it first")
    if (a.out / "audio" / "non-bird").exists(): shutil.rmtree(a.out / "audio" / "non-bird")  # redone whole (screened together)
    rows = species_audio(a.out, xc, a.max_seconds) + non_bird_audio(a.out, fs, a.non_bird, a.screen)

    with (a.out / "ATTRIBUTION.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    (a.out / ".species.csv").unlink(); (a.out / ".species_done").unlink()  # the run is complete: no resume state left
    (a.out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((a.out / 'audio' / r['clip']).read_bytes()).hexdigest()}  "
                                              f"audio/{r['clip']}\n" for r in rows))
    recs = {}
    for r in rows: recs.setdefault(r["species"], set()).add(r["source_id"])
    seconds = sum(float(r["end_s"]) - float(r["start_s"]) for r in rows)
    (a.out / "LICENSE.md").write_text(
        "# Licence of the test clips\n\n"
        "Species clips: stretches of xeno-canto recordings, downmixed to mono, resampled to 32 kHz and re-encoded as 16-bit\n"
        "FLAC; each keeps the licence of its recording (CC0, CC BY or CC BY-SA only), listed with the recordist and the source\n"
        "in `ATTRIBUTION.csv`. Clips from CC BY-SA recordings are redistributed under the same CC BY-SA licence.\n"
        "Non-bird clips: the first 30 s of CC0 Freesound sounds (HQ previews), mono, 32 kHz, 16-bit FLAC; CC0 needs no\n"
        "attribution, the source is listed anyway.\n")
    (a.out / "README.md").write_text(
        "# Test bundle of the bird model\n\n"
        f"{len(rows)} clips, {seconds / 60:.0f} minutes, for measuring the shipped bird model's accuracy "
        "(`python tools/measure_bird_accuracy.py`, specs/006-bird-accuracy-remeasure). Recordings per class: "
        + ", ".join(f"{k} {len(v)}" for k, v in sorted(recs.items())) + ".\n\n"
        "- **Species**: every freely licensed xeno-canto recording of the model's 8 species (CC0, CC BY, CC BY-SA; quality "
        f"A-C), without the stretches the calibration bundle took from it (each calibration clip +-{MARGIN} s), at most "
        f"{a.max_seconds:.0f} s per recording, in stretches of at least {MIN_STRETCH:.0f} s.\n"
        "- **Non-bird**: CC0 Freesound field recordings (" + ", ".join(QUERIES) + "), the first 30 s, kept only if BirdNET "
        f"v2.4 hears no species above {a.screen} in any 3 s window.\n\n"
        "Limits: the species clips come from the same recordings as the calibration clips (other times), and WrenNet was\n"
        "trained on xeno-canto audio, so overlap with its training data cannot be excluded, except for the recordings\n"
        "uploaded after 2025-06-11 (`uploaded` in `ATTRIBUTION.csv`), which the measurement reports separately.\n\n"
        "Built by `tools/make_test_bundle.py`; checked by `SHA256SUMS` and `tests/test_examples.py`. No BioDCASE audio.\n")
    unseen = {r["source_id"] for r in rows if r["source"] == "xeno-canto" and r["uploaded"] > "2025-06-11"}
    print(f"{len(rows)} clips, {seconds / 60:.1f} min; recordings per class:",
          {k: len(v) for k, v in sorted(recs.items())}, f"; unseen recordings: {len(unseen)}")


if __name__ == "__main__":
    main()
