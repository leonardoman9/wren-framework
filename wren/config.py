"""project.yaml: what to convert and how the device should use it. Relative paths are relative to the yaml file."""
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Project:
    name: str
    checkpoint: Path
    classes: list
    negative: str = "last"
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
    firmware_repo: Path = ROOT / "third_party" / "audiomoth-fw"
    tflm: Path = None
    firmware_tests: bool = True
    runs: Path = ROOT / "runs"

    @property
    def run_dir(self):
        return self.runs / self.name


def load(path):
    path = Path(path).resolve()
    base = path.parent
    raw = yaml.safe_load(path.read_text()) or {}
    rel = lambda p: None if p is None else (Path(p) if Path(p).is_absolute() else (base / p).resolve())
    errors = []
    for key in ("name", "checkpoint", "classes"):
        if key not in raw: errors.append(f"missing '{key}'")
    if errors: raise SystemExit(f"{path}: " + "; ".join(errors))
    cal = raw.get("calibration") or {}
    dev = raw.get("device") or {}
    envs = raw.get("envs") or {}
    fw = raw.get("firmware") or {}
    p = Project(
        name=str(raw["name"]), checkpoint=rel(raw["checkpoint"]), classes=[str(c) for c in raw["classes"]],
        negative=str(raw.get("negative", "last")),
        calibration_audio=[rel(a) for a in cal.get("audio", [])], calibration_count=int(cal.get("count", 300)),
        eval_count=int(cal.get("eval", 100)), reference_audio=rel(raw.get("reference_audio")),
        test_audio=[rel(a) for a in (raw.get("test") or {}).get("audio", [])],
        test_windows=int((raw.get("test") or {}).get("windows", 120)),
        window_start=str(dev.get("window_start", "07:00")), window_minutes=int(dev.get("window_minutes", 5)),
        hop_seconds=float(dev.get("hop_seconds", 3.0)), confidence=int(dev.get("confidence", 50)),
    )
    if "torch" in envs: p.torch_python = rel(envs["torch"])
    if "tf" in envs: p.tf_python = rel(envs["tf"])
    if "repo" in fw: p.firmware_repo = rel(fw["repo"])
    p.tflm = rel(fw["tflm"]) if "tflm" in fw else p.firmware_repo / "third_party" / "tflite-micro"
    p.firmware_tests = bool(fw.get("tests", True))
    if "runs" in raw: p.runs = rel(raw["runs"])
    # checks that do not need the model: fail in a second, not after an hour
    if not re.fullmatch(r"[A-Za-z0-9._-]+", p.name): errors.append("name: letters, digits, . _ - only")
    if not p.checkpoint.exists(): errors.append(f"checkpoint not found: {p.checkpoint}")
    if not p.calibration_audio: errors.append("calibration.audio: at least one audio file or folder")
    errors += [f"calibration audio not found: {a}" for a in p.calibration_audio if not a.exists()]
    errors += [f"test audio not found: {a}" for a in p.test_audio if not a.exists()]
    if len(p.classes) < 2: errors.append("classes: at least 2")
    if len(set(p.classes)) != len(p.classes): errors.append("classes: names must be unique")
    if p.calibration_count < 1: errors.append("calibration.count must be positive")
    if errors: raise SystemExit(f"{path}:\n  " + "\n  ".join(errors))
    if not p.test_audio: p.test_audio = p.calibration_audio
    return p
