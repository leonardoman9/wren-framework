"""Builds the bird model's calibration bundle from freely licensed xeno-canto recordings (maintainer tool, run once;
specs/001-reproducible-from-zero/contracts/calibration-bundle.md). The committed bundle is the reference afterwards:
this tool does not need to reproduce it byte for byte.

Only about 1 % of xeno-canto is CC0 / CC-BY / CC-BY-SA, so every usable recording of the model's 8 species is taken
first and the bundle is filled with other European species (FILL_SPECIES). Calibration needs bird audio like the
field's, not the model's species. NC and ND licences are never used (ND forbids cutting).

Usage: XC_API_KEY=... .venv-torch/bin/python tools/make_calibration_bundle.py OUT_DIR [--clips 400] [--test 120]
                                                                              [--per-file 2] [--seed 0]"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
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

ROOT = Path(__file__).resolve().parents[1]
API = "https://xeno-canto.org/api/3/recordings"
SR, WIN, HOP = 32000, 96000, 16000  # 3 s windows on a 0.5 s grid
QUALITY = ("A", "B", "C")
# the rest of the project's 70-species list (the model's 8 species come from examples/birds/project.yaml)
FILL_SPECIES = """Accipiter gentilis, Aegithalos caudatus, Aegolius funereus, Ardea cinerea, Asio otus, Athene noctua,
Buteo buteo, Caprimulgus europaeus, Carduelis carduelis, Chloris chloris, Coccothraustes coccothraustes, Columba palumbus,
Corvus corax, Cuculus canorus, Curruca communis, Cyanistes caeruleus, Delichon urbicum, Dendrocopos major, Dryobates minor,
Dryocopus martius, Emberiza cirlus, Erithacus rubecula, Falco tinnunculus, Ficedula hypoleuca, Fringilla coelebs,
Garrulus glandarius, Hirundo rustica, Jynx torquilla, Lanius collurio, Milvus milvus, Motacilla alba, Motacilla cinerea,
Muscicapa striata, Oriolus oriolus, Otus scops, Parus major, Passer montanus, Pernis apivorus, Phoenicurus ochruros,
Phoenicurus phoenicurus, Phylloscopus bonelli, Phylloscopus collybita, Phylloscopus sibilatrix, Picus canus, Picus viridis,
Poecile palustris, Prunella modularis, Pyrrhula pyrrhula, Regulus ignicapilla, Regulus regulus, Serinus serinus,
Sitta europaea, Spinus spinus, Strix aluco, Sturnus vulgaris, Sylvia atricapilla, Troglodytes troglodytes, Turdus merula,
Turdus philomelos, Turdus pilaris, Turdus viscivorus, Upupa epops"""
FILL_SPECIES = [s.strip() for s in FILL_SPECIES.replace("\n", " ").split(",")]


def licence(url):
    """('CC BY-SA 4.0', canonical URL) for a free licence, None for NC, ND or anything else."""
    url = "https:" + url if url.startswith("//") else url
    if "-nc" in url or "-nd" in url: return None
    m = re.search(r"/licenses/(by|by-sa)/([\d.]+)/", url)
    if m: return f"CC {m.group(1).upper()} {m.group(2)}", url
    m = re.search(r"/publicdomain/zero/([\d.]+)/", url)
    return (f"CC0 {m.group(1)}", url) if m else None


def query(species, key):
    """The freely licensed recordings of one species: the API filters by licence (lic:pd is CC0), all pages, 1 s
    between requests. licence() checks every result again."""
    genus, sp = species.split()
    out = []
    for lic in ("pd", "by", "by-sa"):
        page, pages = 1, 1
        while page <= pages:
            q = urllib.parse.urlencode({"query": f"gen:{genus} sp:{sp} lic:{lic}", "key": key, "page": page})
            d = json.load(urllib.request.urlopen(f"{API}?{q}", timeout=60))
            pages = d["numPages"]
            out += d["recordings"]
            page += 1
            time.sleep(1)
    return out


def usable(recs):
    return [r for r in recs if r.get("q") in QUALITY and licence(r.get("lic", ""))]


def windows(path, per_file):
    """Up to per_file non-overlapping 3 s windows with the highest RMS, as int16 at 32 kHz, with their start (s)."""
    data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    w = torchaudio.functional.resample(torch.from_numpy(data.T.copy()).mean(0, keepdim=True), sr, SR)[0].numpy()
    if len(w) < WIN: return []
    starts = list(range(0, len(w) - WIN + 1, HOP))
    rms = [float(np.sqrt(np.mean(w[s:s + WIN] ** 2))) for s in starts]
    picked = []
    for i in np.argsort(rms)[::-1]:
        s = starts[i]
        if rms[i] < 1e-4 or len(picked) == per_file: break
        if all(abs(s - p) >= WIN for p in picked): picked.append(s)
    return [(s / SR, np.clip(np.rint(w[s:s + WIN] * 32768.0), -32768, 32767).astype(np.int16)) for s in sorted(picked)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--clips", type=int, default=400, help="calibration + evaluation windows (calibration.count + eval)")
    ap.add_argument("--test", type=int, default=120, help="extra windows for the device test (test.windows)")
    ap.add_argument("--per-file", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if "biodcase" in str(a.out.resolve()).lower(): sys.exit("refusing: BioDCASE must not be used anywhere")
    key = os.environ.get("XC_API_KEY") or sys.exit("set XC_API_KEY (xeno-canto API key)")
    model = [c for c in yaml.safe_load((ROOT / "examples/birds/project.yaml").read_text())["classes"] if " " in c]
    need = a.clips + a.test

    # model species first (every usable recording), then fill species round-robin, each shuffled with the seed
    rng = random.Random(a.seed)
    found = {}
    for sp in model + FILL_SPECIES:
        recs = sorted(usable(query(sp, key)), key=lambda r: int(r["id"]))
        rng.shuffle(recs)
        found[sp] = recs
        print(f"{sp}: {len(recs)} usable", flush=True)
    order = [r for sp in model for r in found[sp]]
    fill = [list(found[sp]) for sp in FILL_SPECIES]
    while any(fill):
        order += [q.pop(0) for q in fill if q]

    audio = a.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rows, done = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for r in order:
            if done >= need: break
            src = Path(tmp) / f"XC{r['id']}"
            try:
                urllib.request.urlretrieve(r["file"], src)
                cut = windows(src, a.per_file)
            except Exception as e:  # unreadable or unreachable: skip, the bundle is filled from the next ones
                print(f"  skip XC{r['id']}: {e}"); continue
            finally:
                src.unlink(missing_ok=True)
                time.sleep(0.5)
            name, url = licence(r["lic"])
            for start, pcm in cut[:need - done]:
                clip = f"XC{r['id']}_{int(start):04d}.flac"
                soundfile.write(str(audio / clip), pcm, SR, subtype="PCM_16", format="FLAC")
                rows.append({"clip": clip, "xc_id": r["id"], "species": f"{r['gen']} {r['sp']}", "recordist": r["rec"],
                             "licence": name, "licence_url": url, "source_url": "https://xeno-canto.org/" + r["id"],
                             "start_s": f"{start:.1f}"})
                done += 1
            print(f"  {done}/{need}", flush=True)
    if done < need: sys.exit(f"only {done} windows from freely licensed recordings, {need} wanted")

    with (a.out / "ATTRIBUTION.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (a.out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((audio / r['clip']).read_bytes()).hexdigest()}  audio/{r['clip']}\n"
                                              for r in rows))
    n_model = sum(r["species"] in model for r in rows)
    (a.out / "LICENSE.md").write_text(
        "# Licence of the calibration clips\n\n"
        "Every clip in `audio/` is a 3 s excerpt of a xeno-canto recording, cut, downmixed to mono, resampled to 32 kHz\n"
        "and re-encoded as 16-bit FLAC. Each clip keeps the licence of its recording, listed with the recordist and the\n"
        "source in `ATTRIBUTION.csv`. Only CC0, CC BY and CC BY-SA recordings were used. Clips from CC BY-SA recordings\n"
        "are redistributed under the same CC BY-SA licence.\n")
    (a.out / "README.md").write_text(
        "# Calibration bundle of the bird model\n\n"
        f"{len(rows)} clips of 3 s (mono, 32 kHz, 16-bit FLAC) from freely licensed xeno-canto recordings (API v3): "
        f"{n_model} from the model's 8 species, {len(rows) - n_model} from other European species, which complete it "
        "because only about 1 % of xeno-canto is freely licensed. Up to "
        f"{a.per_file} windows per recording, the loudest non-overlapping ones (so that they contain calls), quality "
        f"{'/'.join(QUALITY)}, seed {a.seed}.\n\n"
        "The pipeline uses it for the int8 calibration and evaluation (`calibration.audio`) and for the device test windows\n"
        "(`test.audio`) of `examples/birds/project.yaml`. Changing any clip changes the int8 model and the test vector.\n"
        "Built by `tools/make_calibration_bundle.py`; checked by `SHA256SUMS` and `tests/test_examples.py`.\n")
    print(f"{len(rows)} clips ({n_model} from the model's species) in {a.out}")


if __name__ == "__main__":
    main()
