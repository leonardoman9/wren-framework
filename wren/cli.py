"""wren: from a trained WrenNet checkpoint to AudioMoth firmware.

  python -m wren setup [--tflm]          the two Python environments (torch, TensorFlow) and, with --tflm, TFLM for the
                                         firmware (fetched at the verified commit and built; ~10 minutes the first time)
  python -m wren run PROJECT.yaml [--stages export,package,firmware,report] [--force]
"""
import argparse
import json
import shutil
import subprocess
import sys

from wren import config
from wren.runner import Runner, StepFailed
from wren.stages import data, export, firmware, package, report

STAGES = ["prepare", "train", "evaluate", "export", "package", "firmware", "report"]


def setup(args):
    py = args.python or shutil.which("python3.11")
    if not py: raise SystemExit("Python 3.11 is needed for the environments (pass --python PATH)")
    for name in ("torch", "tf"):
        venv = config.ROOT / f".venv-{name}"
        if not (venv / "bin" / "python").exists():
            print(f"creating {venv}")
            subprocess.run([py, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "install", "-q", "-r",
                        str(config.ROOT / "envs" / f"requirements-{name}.txt")], check=True)
    if args.tflm:
        fw = config.ROOT / "third_party" / "audiomoth-fw"
        subprocess.run(["sh", str(fw / "scripts" / "setup_tflm.sh")], check=True)
        subprocess.run(["sh", str(fw / "scripts" / "build_tflm.sh")], check=True, env={**__import__("os").environ, "HOST": "1"})
    print("ready")


def run(args):
    p = config.load(args.project)
    stages = args.stages.split(",") if args.stages else STAGES
    bad = [s for s in stages if s not in STAGES]
    if bad: raise SystemExit(f"unknown stages {bad}; stages: {STAGES}")
    for env, py in (("torch", p.torch_python), ("tf", p.tf_python)):
        if not py.exists(): raise SystemExit(f"{env} environment missing ({py}): run `python -m wren setup`")
    r = Runner(p)
    source = f"audio in {', '.join(p.data['sources'].values())}" if p.trains else p.checkpoint.name
    print(f"{p.name}: {source}, {len(p.classes)} classes -> {p.run_dir}")
    try:
        if p.trains:
            done_key = lambda d: json.loads((d / "stage.json").read_text())["key"] if (d / "stage.json").exists() else None
            prep, pkey = data.prepare(r, args.force) if "prepare" in stages else (p.run_dir / "prepare", done_key(p.run_dir / "prepare"))
            tkey = data.train(r, prep, pkey, args.force)[1] if "train" in stages else done_key(p.run_dir / "train")
            if "evaluate" in stages: data.evaluate(r, prep, tkey, args.force)
            if not p.checkpoint.exists(): raise StepFailed(f"no trained model yet ({p.checkpoint}): run the train stage")
        exp = export.run(r, args.force) if "export" in stages else p.run_dir / "export"
        pkg = package.run(r, exp, args.force) if "package" in stages else p.run_dir / "package"
        if "firmware" in stages: firmware.run(r, pkg, args.force)
        if "report" in stages: report.run(r)
    except StepFailed as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(prog="wren", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="create the environments")
    s.add_argument("--tflm", action="store_true", help="also fetch and build TFLM for the firmware")
    s.add_argument("--python", help="Python 3.11 interpreter for the environments")
    s.set_defaults(fn=setup)
    r = sub.add_parser("run", help="run the pipeline on a project.yaml")
    r.add_argument("project")
    r.add_argument("--stages", help="comma-separated subset of " + ",".join(STAGES) + " (prepare/train/evaluate: training projects)")
    r.add_argument("--force", action="store_true", help="rerun stages even if their inputs did not change")
    r.set_defaults(fn=run)
    a = ap.parse_args()
    a.fn(a)
