"""project.yaml: what to convert (a checkpoint) or train (audio folders), and how the device should use it.
Relative paths are relative to the yaml file; without 'name', the project is named after its folder. Checks that need
no model run here, so mistakes cost seconds."""
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAX_CLASSES = 180  # what the firmware's flash holds (~88 B per class)


@dataclass
class Project:
    name: str
    checkpoint: Path                # given (convert) or runs/<name>/train/best.pt (train)
    classes: list                   # display names, in the model's output order
    negative: str = "last"
    data: dict = None               # training projects: sources, negative folder, split, windows (see README.md, project.yaml)
    class_folders: list = None      # training projects: folder names, same order as classes
    training: dict = field(default_factory=dict)
    calibration_audio: list = field(default_factory=list)
    calibration_count: int = 300
    eval_count: int = 100
    reference_audio: Path = None
    test_audio: list = field(default_factory=list)
    test_windows: int = 120
    window_start: str = "07:00"
    window_minutes: int = 5
    hop_seconds: float = 3.0
    confidence: int = 50
    torch_python: Path = ROOT / ".venv-torch" / "bin" / "python"
    tf_python: Path = ROOT / ".venv-tf" / "bin" / "python"
    firmware_repo: Path = ROOT / "audiomoth"
    tflm: Path = None
    firmware_tests: bool = True
    runs: Path = ROOT / "runs"

    @property
    def run_dir(self):
        return self.runs / self.name

    @property
    def trains(self):
        return self.data is not None

    @property
    def negative_index(self):
        if self.negative == "none": return -1
        return len(self.classes) - 1 if self.negative == "last" else int(self.negative)


TRAINING_DEFAULTS = {"init": "wrennet", "epochs": 40, "batch_size": 32, "lr": 5.0e-4, "negative_share": 0.4, "loss": "ce",
                     "focal_gamma": 2.0, "augment": True, "seed": 0, "device": "auto", "patience": 8, "warmup_epochs": 2}
# learning from BirdNET (specs/003-birdnet-distillation): off unless asked; only in p.training when on, so projects that
# do not use it train exactly as before
BIRDNET_DEFAULTS = {"birdnet": False, "birdnet_weight": 0.5, "birdnet_temperature": 4.0, "birdnet_species": {}}


KNOWN = {  # every key project.yaml may contain: a misspelled key must not silently fall back to a default
    "": {"name", "checkpoint", "classes", "negative", "data", "training", "calibration", "reference_audio", "test", "device",
         "envs", "firmware", "runs"},
    "data": {"root", "train", "validation", "test", "negative", "names", "group", "split", "windows"},
    "data.split": {"validation", "test", "seed"},
    "data.windows": {"hop_seconds", "max_per_file", "negative_max_per_file"},
    "training": set(TRAINING_DEFAULTS) | set(BIRDNET_DEFAULTS),
    "calibration": {"audio", "count", "eval"},
    "test": {"audio", "windows"},
    "device": {"window_start", "window_minutes", "hop_seconds", "confidence"},
    "envs": {"torch", "tf"},
    "firmware": {"repo", "tflm", "tests"},
}


def unknown_keys(raw):
    problems = []
    for where, allowed in KNOWN.items():
        node = raw
        for part in [q for q in where.split(".") if q]:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            problems += [f"unknown key '{(where + '.') if where else ''}{k}' (known: {', '.join(sorted(allowed))})" for k in node if k not in allowed]
    return problems


def name_problem(n):
    """The firmware's rules for class names (gen_model_config.py): 1-40 printable ASCII, no quotes, backslashes, commas."""
    if not 1 <= len(n) <= 40: return "1-40 characters"
    if not all(32 <= ord(c) < 127 for c in n): return "printable ASCII only"
    if any(c in n for c in '"\\,'): return "no quotes, backslashes or commas"
    return None


def birdnet_checks(p, given, folders, negative):
    """Learning from BirdNET (specs/003-birdnet-distillation, contracts/project-yaml.md): only for bird species BirdNET
    knows. Fills p.training's birdnet keys and the matched species; returns the problems, in plain words."""
    from wren import birdnet
    b = {**BIRDNET_DEFAULTS, **{k: v for k, v in given.items() if k in BIRDNET_DEFAULTS}}
    errors = []
    if "loss" in given and given["loss"] != "focal":
        errors.append("Learning from BirdNET uses the focal loss; remove 'loss: ce' (or set birdnet: false).")
    p.training["loss"] = "focal"
    try: weight, temperature = float(b["birdnet_weight"]), float(b["birdnet_temperature"])
    except (TypeError, ValueError): weight, temperature = -1.0, 0.0
    if not 0 <= weight <= 1: errors.append("training.birdnet_weight: a number from 0 to 1")
    if not temperature > 0: errors.append("training.birdnet_temperature: a number above 0")
    mapping = b["birdnet_species"] or {}
    sounds = [f for f in folders if f != negative]
    errors += [f"training.birdnet_species: '{k}' is not one of your sound folders" for k in mapping if k not in sounds]
    if errors: return errors
    if birdnet.ready():
        matched, unmatched = birdnet.match(folders, negative, mapping, birdnet.labels())
        if unmatched:
            return ["Learning from BirdNET works only for bird species BirdNET knows. Not found: "
                    + ", ".join(f"'{f}'" for f in unmatched) + ". Name each folder after its scientific name, or give it "
                    f"under training: birdnet_species (e.g. {unmatched[0]}: Turdus merula)."]
    else:
        stored = birdnet.stored_mapping(p.run_dir)
        if stored is None:
            return ["Learning from BirdNET needs BirdNET on this computer: run python -m wren setup --birdnet"]
        missing = [f for f in sounds if f not in stored or stored[f]["name"].lower() != birdnet.scientific(f, mapping).lower()]
        if missing:
            return ["BirdNET's stored results do not cover " + ", ".join(f"'{f}'" for f in missing)
                    + ": run python -m wren setup --birdnet"]
        matched = {f: stored[f] for f in sounds}
    p.training.update(birdnet=True, birdnet_weight=weight, birdnet_temperature=temperature,
                      birdnet_species={f: matched[f] for f in sounds})
    return []


def default_name(folder):
    """A project without 'name' is named after its folder (runs/<folder>), in the characters a name may use."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", folder.name).strip("-") or "project"


def load(path):
    path = Path(path).resolve()
    base = path.parent
    raw = yaml.safe_load(path.read_text()) or {}
    rel = lambda p: None if p is None else (Path(p) if Path(p).is_absolute() else (base / p).resolve())
    errors = unknown_keys(raw)
    if ("checkpoint" in raw) == ("data" in raw): errors.append("give either 'checkpoint' (convert a model) or 'data' (train one)")
    if errors: raise SystemExit(f"{path}: " + "; ".join(errors))
    cal, dev, envs, fw, test = (raw.get(k) or {} for k in ("calibration", "device", "envs", "firmware", "test"))
    p = Project(
        name=str(raw["name"]) if "name" in raw else default_name(base), checkpoint=None, classes=[],
        calibration_audio=[rel(a) for a in cal.get("audio", [])], calibration_count=int(cal.get("count", 300)),
        eval_count=int(cal.get("eval", 100)), reference_audio=rel(raw.get("reference_audio")),
        test_audio=[rel(a) for a in test.get("audio", [])], test_windows=int(test.get("windows", 120)),
        window_start=str(dev.get("window_start", "07:00")), window_minutes=int(dev.get("window_minutes", 5)),
        hop_seconds=float(dev.get("hop_seconds", 3.0)), confidence=int(dev.get("confidence", 50)),
    )
    if "runs" in raw: p.runs = rel(raw["runs"])
    if "data" in raw:
        d = raw["data"] or {}
        sources = {k: rel(d[k]) for k in ("root", "train", "validation", "test") if d.get(k)}
        if ("root" in sources) == ("train" in sources): errors.append("data: give either 'root' or 'train' (+ optional validation, test)")
        listing = sources.get("root") or sources.get("train")
        folders = sorted(q.name for q in listing.iterdir() if q.is_dir() and not q.name.startswith(".")) if listing and listing.is_dir() else []
        if listing and not listing.is_dir(): errors.append(f"data folder not found: {listing}")
        negative = d.get("negative")
        if negative is not None and negative not in folders: errors.append(f"data.negative: no folder '{negative}' in {listing}")
        folders = [f for f in folders if f != negative] + ([negative] if negative in folders else [])
        names = d.get("names") or {}
        errors += [f"data.names: no class folder '{k}'" for k in names if k not in folders]
        p.class_folders, p.classes = folders, [str(names.get(f, f)) for f in folders]
        if negative is None: print("note: no data.negative: the model will always report one of the classes (no 'nothing' class)")
        p.negative = "last" if negative is not None else "none"
        p.data = {"sources": {k: str(v) for k, v in sources.items()}, "group": d.get("group", "file"),
                  "split": {"validation": 0.15, "test": 0.15, "seed": 0, **(d.get("split") or {})},
                  "windows": {"hop_seconds": 1.5, "max_per_file": 6, "negative_max_per_file": 40, **(d.get("windows") or {})}}
        if p.data["group"] not in ("file", "subfolder"): errors.append("data.group: file or subfolder")
        given = raw.get("training") or {}
        p.training = {**TRAINING_DEFAULTS, **{k: v for k, v in given.items() if k not in BIRDNET_DEFAULTS}}
        if p.training["init"] not in ("wrennet", "scratch"): errors.append("training.init: wrennet or scratch")
        if p.training["loss"] not in ("ce", "focal"): errors.append("training.loss: ce or focal")
        if given.get("birdnet"): errors += birdnet_checks(p, given, folders, negative)
        p.checkpoint = p.run_dir / "train" / "best.pt"
    else:
        p.checkpoint = rel(raw["checkpoint"])
        if any(k in BIRDNET_DEFAULTS for k in raw.get("training") or {}):
            errors.append("training.birdnet: only for training projects (data:)")
        if "classes" not in raw: errors.append("missing 'classes' (names in the checkpoint's output order)")
        p.classes = [str(c) for c in raw.get("classes", [])]
        p.negative = str(raw.get("negative", "last"))
        if not p.checkpoint.exists(): errors.append(f"checkpoint not found: {p.checkpoint}")
        if not p.calibration_audio: errors.append("calibration.audio: at least one audio file or folder")
        errors += [f"calibration audio not found: {a}" for a in p.calibration_audio if not a.exists()]
        errors += [f"test audio not found: {a}" for a in p.test_audio if not a.exists()]
        if not p.test_audio: p.test_audio = p.calibration_audio
    if "torch" in envs: p.torch_python = rel(envs["torch"])
    if "tf" in envs: p.tf_python = rel(envs["tf"])
    if "repo" in fw: p.firmware_repo = rel(fw["repo"])
    p.tflm = rel(fw["tflm"]) if "tflm" in fw else p.firmware_repo / "third_party" / "tflite-micro"
    p.firmware_tests = bool(fw.get("tests", True))
    if not re.fullmatch(r"[A-Za-z0-9._-]+", p.name): errors.append("name: letters, digits, . _ - only")
    if not 2 <= len(p.classes) <= MAX_CLASSES: errors.append(f"{len(p.classes)} classes: 2-{MAX_CLASSES} are possible")
    if len(set(p.classes)) != len(p.classes): errors.append("class names must be unique")
    for c in p.classes:
        if name_problem(c): errors.append(f"class name '{c}': {name_problem(c)} (rename it with data.names)")
    if p.calibration_count < 1: errors.append("calibration.count must be positive")
    if errors: raise SystemExit(f"{path}:\n  " + "\n  ".join(errors))
    return p
