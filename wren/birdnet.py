"""BirdNET, the teacher of `training: birdnet: true` (specs/003-birdnet-distillation): where its files live, which files
they must be, and how a project's sounds are matched to its species. Standard library only: used by setup and by the
project check, before any environment exists.

BirdNET v2.4 GLOBAL 6K, the official TFLite release (Zenodo 15050749), licensed CC BY-NC 4.0; fetched by
`python -m wren setup --birdnet`, never committed."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "third_party" / "birdnet" / "v2.4"
ZIP_URL = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip?download=1"
ZIP_SHA256 = "31377e128d86fe7b65fa91b206b8804bad1cd934e624bbe6e7c1788170c95e57"
FILES = {"audio-model.tflite": "55f3e4055b1a13bfa9a2452731d0d34f6a02d6b775a334362665892794165e4c",
         "labels/en_us.txt": "b50b77b7c3dfe40cd637e8cccdca0173a0a4ddee8867b830ff3c1a566f477f16"}
MARKER = ".wren-birdnet"
MODEL_NAME = "BirdNET v2.4 GLOBAL 6K (Zenodo 15050749)"
LICENCE = "CC BY-NC 4.0"
CITATION = "Kahl et al. (2021), BirdNET: A deep learning solution for avian diversity monitoring, Ecological Informatics 61:101236"


def ready(d=None):
    """True when both files are installed and the marker (written last by setup) lists their pinned hashes."""
    d = Path(d or DIR)
    marker = d / MARKER
    return marker.exists() and json.loads(marker.read_text()) == FILES and all((d / f).exists() for f in FILES)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def labels(path=None):
    """Scientific names in BirdNET's output order (each line is 'Scientific name_Common name')."""
    return [line.split("_", 1)[0] for line in Path(path or DIR / "labels" / "en_us.txt").read_text().splitlines()]


def scientific(folder, mapping):
    return str(mapping.get(folder) or folder.replace("_", " ")).strip()


def match(folders, negative, mapping, names):
    """Each sound folder except the background -> (scientific name, BirdNET index). Returns (matched, unmatched): the
    name must equal, ignoring case, exactly one label's scientific name."""
    index = {}
    for i, n in enumerate(names):
        index.setdefault(n.lower(), []).append(i)
    matched, unmatched = {}, []
    for f in folders:
        if f == negative: continue
        name = scientific(f, mapping)
        hits = index.get(name.lower(), [])
        if len(hits) == 1: matched[f] = {"name": names[hits[0]], "index": hits[0]}
        else: unmatched.append(f)
    return matched, unmatched


def stored_mapping(run_dir):
    """The species mapping a previous run stored with BirdNET's results, or None (lets a computer without BirdNET
    repeat the training)."""
    f = Path(run_dir) / "train" / "birdnet" / "mapping.json"
    return json.loads(f.read_text())["species"] if f.exists() else None
