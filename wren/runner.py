"""Running steps: each in its environment (torch, tf or the host), logged to runs/<name>/logs/, stopping on errors."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from wren.config import ROOT


class StepFailed(SystemExit):
    pass


# what each step does, in the words printed to the user (constitution I); tool names stay in logs/<step>.log
LABELS = {
    "prepare": "reading and splitting the audio", "train": "training the model",
    "evaluate": "testing the model on audio it has not seen", "reference": "computing the original model's results",
    "export_backbone": "exporting the model", "onnxsim": "simplifying the model",
    "onnx2tf": "converting the model for the device", "make_specs": "preparing the calibration audio",
    "quantize": "compressing the model for the device", "frontend_tables_build": "preparing the sound filter",
    "frontend_tables": "computing the sound filter", "tail_weights": "writing the model's last layers",
    "model_config": "writing the class list and device settings", "libfe_build": "building the device code to test it here",
    "libtail_build": "building the device code to test it here", "test_reference": "computing results to compare with",
    "pipeline_check": "checking the device version against the original model",
    "device_eval": "measuring the device version on the test audio", "test_vectors": "writing the test window",
    "test_spec": "writing the test window", "build": "building the firmware", "build_diag": "building the diagnostic firmware",
    "tests": "testing the firmware on a simulated device (a few minutes)",
    "birdnet_windows": "preparing the training audio for BirdNET",
    "birdnet": "asking BirdNET what it hears in the training audio",
}


# The stages that must give the same bytes on every x86 computer (export, package) run their numerical libraries on
# one fixed code path: their fastest kernels are chosen from the CPU at run time (AVX-512, AVX2, emulated AVX2 on
# Apple silicon...) and from the thread count, and sum in different orders. Training is not pinned: it is judged by
# the quality gates, not by bytes (constitution V).
PINNED = {"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE,STRICT", "MKL_ENABLE_INSTRUCTIONS": "SSE4_2",
          "ONEDNN_MAX_CPU_ISA": "SSE41",
          "DNNL_MAX_CPU_ISA": "SSE41", "TF_ENABLE_ONEDNN_OPTS": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
          "OPENBLAS_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1"}


class Runner:
    def __init__(self, project):
        self.p = project
        self.pinned = False  # set by the export and package stages
        self.logs = project.run_dir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def run(self, name, cmd, env=None, cwd=None):
        """Runs cmd, writes its output to logs/<name>.log, returns the output; raises StepFailed on a non-zero exit."""
        e = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                 **(PINNED if self.pinned else {}), **(env or {}))
        cmd = [str(c) for c in cmd]
        label = LABELS.get(name, name.replace("_", " "))
        print(f"  - {label}", flush=True)
        r = subprocess.run(cmd, env=e, cwd=cwd, capture_output=True, text=True)
        log = self.logs / f"{name}.log"
        log.write_text("$ " + " ".join(cmd) + "\n" + r.stdout + r.stderr)
        if r.returncode != 0:
            raise StepFailed(f"{label[0].upper() + label[1:]} did not work. The details are in {log}; please include "
                             "that file if you ask for help.")
        return r.stdout

    def torch(self, name, *args):
        return self.run(name, [self.p.torch_python, *args])

    def tf(self, name, *args):
        return self.run(name, [self.p.tf_python, *args])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_key(*parts):
    """Hash of a stage's inputs: files by content, everything else by value."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, Path) and p.is_file(): h.update(sha256(p).encode())
        else: h.update(json.dumps(p, default=str, sort_keys=True).encode())
    return h.hexdigest()


def listing(*folders):
    """Every file under the folders with its size and modification time: a stage key that notices added, removed or
    changed audio (hashing paths alone would not)."""
    items = []
    for d in folders:
        d = Path(d)
        if d.is_file(): items.append((str(d), d.stat().st_size, int(d.stat().st_mtime)))
        elif d.is_dir():
            items += sorted((str(f), f.stat().st_size, int(f.stat().st_mtime)) for f in d.rglob("*") if f.is_file())
    return items


def model_code(root=ROOT):
    """Hash of the model's source: WrenNet's architecture (wrennet/**/*.py) and wren's loader (wren/torchside/model.py).
    Part of the key of every stage that loads the model (specs/004-model-code-stage-keys), so a change to the model code
    re-runs them; READMEs and caches do not count, and the checkpoint enters the keys by content where it is used."""
    root = Path(root)
    files = sorted((root / "wrennet").rglob("*.py")) + [root / "wren" / "torchside" / "model.py"]
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(root)).encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def code(*names):
    """Hash of wren's own code files, so a changed stage reruns."""
    return [sha256(ROOT / "wren" / n) for n in names]


SKIP_DIRS = {"__pycache__", "third_party", "out"}  # build output and fetched code, as in audiomoth/.gitignore


def tree_hash(folder):
    """Hash of the source files under a folder (the WrenNet architecture, the firmware): part of the keys that depend on
    it. Build output (out/, out_*/), fetched code (third_party/) and caches are left out."""
    files = []
    for d, subdirs, names in os.walk(folder):
        subdirs[:] = [s for s in subdirs if s not in SKIP_DIRS and not s.startswith("out_")]
        files += [Path(d) / n for n in names if not n.endswith(".pyc") and n != ".DS_Store"]
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(str(f.relative_to(folder)).encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def content(*paths):
    """Content hashes of files and folders (audio inputs): part of a stage key, so changed clips rerun the stage."""
    return [tree_hash(p) if Path(p).is_dir() else sha256(p) if Path(p).is_file() else None for p in paths if p]


def up_to_date(out_dir, key):
    f = Path(out_dir) / "stage.json"
    return f.exists() and json.loads(f.read_text()).get("key") == key


def mark_done(out_dir, key, **info):
    (Path(out_dir) / "stage.json").write_text(json.dumps({"key": key, **info}, indent=2) + "\n")


def need(path, what):
    if not Path(path).exists(): raise StepFailed(f"{what} not found: {path}")
    return Path(path)


def host_python():
    return sys.executable
