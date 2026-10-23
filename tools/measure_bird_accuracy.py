"""Measures the shipped bird model's accuracy on the test bundle examples/birds/test/ (specs/006-bird-accuracy-remeasure,
contracts/measurement.md): the pipeline's own call-finding chooses the windows, PyTorch gives the float model's answers
and the firmware's own code (C front-end + the shipped int8 backbone + C tail) the device's.
Usage: python tools/measure_bird_accuracy.py [OUT_DIR]      (default runs/bird-accuracy; writes accuracy.json, accuracy.md)

Every step runs with the export and package stages' fixed CPU settings (wren.runner.PINNED), so in the Docker image the
result is the same bytes on any computer (tests/test_bird_accuracy.sh)."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import yaml
except ImportError:  # the typed Python lacks the pipeline's packages: re-run in the torch environment, as python -m wren
    venv = ROOT / ".venv-torch" / "bin" / "python"
    if not venv.exists(): sys.exit("Not installed yet: run `python -m wren setup --tflm` first.")
    os.execv(venv, [str(venv), *sys.argv])
sys.path.insert(0, str(ROOT))
from wren.runner import PINNED  # noqa: E402

BUNDLE = ROOT / "examples" / "birds" / "test"
PACKAGE = ROOT / "audiomoth" / "firmware" / "bird" / "model"
BIRD = ROOT / "audiomoth" / "firmware" / "bird"
CHECKPOINT = ROOT / "wrennet" / "checkpoints" / "birds.pt"
UNSEEN_AFTER = "2025-06-11"  # WrenNet's checkpoint first committed to sciapponi/WrenNet (df95736): research R3
TORCH, TF = ROOT / ".venv-torch" / "bin" / "python", ROOT / ".venv-tf" / "bin" / "python"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def step(name, cmd, logs):
    env = dict(os.environ, PYTHONPATH=str(ROOT), **PINNED)
    r = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, cwd=ROOT)
    (logs / f"{name}.log").write_text("$ " + " ".join(str(c) for c in cmd) + "\n" + r.stdout + r.stderr)
    if r.returncode != 0: sys.exit(f"The measurement failed at '{name}'. The details are in {logs / (name + '.log')}.")
    return r.stdout


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "bird-accuracy"
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    sums = BUNDLE / "SHA256SUMS"
    if not sums.exists(): sys.exit(f"The test bundle is missing: {BUNDLE} (built by tools/make_test_bundle.py).")
    bad = [p for h, p in (line.split("  ", 1) for line in sums.read_text().splitlines()) if sha256(BUNDLE / p) != h]
    if bad: sys.exit(f"{len(bad)} test files differ from {sums}, e.g. {bad[0]}: the bundle is not the committed one.")

    project = yaml.safe_load((ROOT / "examples" / "birds" / "project.yaml").read_text())
    classes = project["classes"]
    spec = {"classes": [c.replace(" ", "_") for c in classes], "negative": len(classes) - 1,
            "sources": {"train": str(BUNDLE / "audio")}, "split": {"validation": 0, "test": 0, "seed": 0},
            "group": "subfolder", "windows": {"hop_seconds": 1.5, "max_per_file": 6, "negative_max_per_file": 40}}
    (out / "prepare_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print("finding the calls in the test recordings", flush=True)
    step("prepare", [TORCH, "-m", "wren.torchside.prepare", out / "prepare_spec.json", out / "prepare"], logs)
    print("computing the original model's answers", flush=True)
    step("windows_ref", [TORCH, "-m", "wren.torchside.windows_ref", out / "prepare", "train", CHECKPOINT, out / "ref.npz",
                         1_000_000, 0], logs)
    print("building the device code to run it here", flush=True)
    for name, src in (("libfe", "bird_frontend.c"), ("libtail", "bird_tail.c")):
        step(f"{name}_build", ["cc", "-O2", "-shared", "-fPIC", f"-I{BIRD}", f"-I{PACKAGE / 'generated'}", BIRD / src, "-lm",
                               "-o", out / f"{name}.so"], logs)
    print("running the device version on every window", flush=True)
    step("device_eval", [TF, "-m", "wren.tfside.device_eval", out / "ref.npz", PACKAGE / "bird_backbone_int8.tflite",
                         out / "libfe.so", out / "libtail.so", len(classes), out / "device_eval.json", out / "windows.npz"], logs)
    info = {"classes": classes, "negative": len(classes) - 1,
            "confidence_threshold": project["device"]["confidence"] / 100, "unseen_after": UNSEEN_AFTER,
            "package_sha256": sha256(PACKAGE / "bird_backbone_int8.tflite"), "checkpoint_sha256": sha256(CHECKPOINT),
            "bundle_sha256": sha256(sums)}
    (out / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    print("computing the figures and their intervals", flush=True)
    print(step("accuracy", [TORCH, "-m", "wren.torchside.accuracy", out / "windows.npz", out / "ref.npz",
                            BUNDLE / "ATTRIBUTION.csv", out / "info.json", out], logs))
    print(f"accuracy.json and accuracy.md in {out}")


if __name__ == "__main__":
    main()
