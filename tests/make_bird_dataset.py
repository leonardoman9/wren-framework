"""A small bird project from the committed calibration bundle (examples/birds/calibration: freely licensed xeno-canto
clips, no BioDCASE), for learning from BirdNET (specs/003-birdnet-distillation, research R10).
Usage: python tests/make_bird_dataset.py OUT_DIR      (standard library only)

  OUT/<Genus_species>/XC<id>/<clip>   the bird model's 8 species, at most 10 recordings each (the 10 lowest xeno-canto
                                      ids, so the set is fixed): the "10 or fewer recordings" of the spec's SC-003
  OUT/background/XC<id>/<clip>        the other 62 species' clips: "none of the project's species"
Clips of one recording may end up in different splits (data.group: file): a limit of the comparison, not of the test."""
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SPECIES = ["Bubo bubo", "Certhia familiaris", "Apus apus", "Certhia brachydactyla", "Emberiza cia", "Lophophanes cristatus",
           "Periparus ater", "Poecile montanus"]
MAX_RECORDINGS = 10

bundle = Path(__file__).resolve().parents[1] / "examples" / "birds" / "calibration"
out = Path(sys.argv[1])
if out.exists() and any(out.iterdir()): sys.exit(f"{out} is not empty")
rows = list(csv.DictReader((bundle / "ATTRIBUTION.csv").open()))
recordings = defaultdict(set)
for r in rows: recordings[r["species"]].add(int(r["xc_id"]))
keep = {s: set(sorted(recordings[s])[:MAX_RECORDINGS]) for s in SPECIES}
n = defaultdict(int)
for r in rows:
    s, xc = r["species"], int(r["xc_id"])
    if s in SPECIES and xc not in keep[s]: continue
    folder = s.replace(" ", "_") if s in SPECIES else "background"
    dest = out / folder / f"XC{xc}" / r["clip"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle / "audio" / r["clip"], dest)
    n[folder] += 1
print(", ".join(f"{k}: {v} clips" for k, v in sorted(n.items())))
