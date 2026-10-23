"""wren: from audio folders or a trained WrenNet checkpoint to AudioMoth firmware.

  python -m wren setup [--tflm] [--birdnet] [--check [--verbose]] [--python PATH]
                                         the two Python environments (torch, TensorFlow) and, with --tflm, the firmware
                                         build tools (~10 minutes the first time); --birdnet: BirdNET, for
                                         training: birdnet: true; --check only reports what is ready
  python -m wren run PROJECT.yaml [--stages export,package,firmware,report] [--force]

Only the standard library is imported here: `setup` must work on a bare Python before anything is installed.
"""
import argparse
import json
import os
import sys

STAGES = ["prepare", "train", "evaluate", "export", "package", "firmware", "report"]


def setup(args):
    from wren import setup as wren_setup
    sys.exit(wren_setup.main(tflm=args.tflm, check=args.check, verbose=args.verbose, python=args.python,
                             birdnet_too=args.birdnet))


def run(args):
    try:
        import yaml  # noqa: F401  (present in the environments, not necessarily in the Python that was typed)
    except ImportError:
        from wren.setup import ROOT
        venv_python = ROOT / ".venv-torch" / "bin" / "python"
        if not venv_python.exists(): raise SystemExit("Not installed yet: run `python -m wren setup --tflm` first.")
        os.execv(venv_python, [str(venv_python), "-m", "wren", *sys.argv[1:]])
    from wren import config
    from wren.runner import Runner, StepFailed
    from wren.stages import data, export, firmware, package, report

    p = config.load(args.project)
    stages = args.stages.split(",") if args.stages else STAGES
    bad = [s for s in stages if s not in STAGES]
    if bad: raise SystemExit(f"unknown stages {bad}; stages: {STAGES}")
    for env, py in (("torch", p.torch_python), ("tf", p.tf_python)):
        if not py.exists(): raise SystemExit("Not installed yet: run `python -m wren setup --tflm` first.")
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
    s = sub.add_parser("setup", help="install what the pipeline needs, or check it (--check)")
    s.add_argument("--tflm", action="store_true", help="also fetch and build the firmware build tools (TFLM, compiler)")
    s.add_argument("--birdnet", action="store_true", help="also fetch BirdNET (for training: birdnet: true; ~77 MB)")
    s.add_argument("--check", action="store_true", help="change nothing; report what is ready and what is missing")
    s.add_argument("--verbose", action="store_true", help="with --check: also show component names and versions")
    s.add_argument("--python", help="Python 3.11 interpreter for the environments")
    s.set_defaults(fn=setup)
    r = sub.add_parser("run", help="run the pipeline on a project.yaml")
    r.add_argument("project")
    r.add_argument("--stages", help="comma-separated subset of " + ",".join(STAGES) + " (prepare/train/evaluate: training projects)")
    r.add_argument("--force", action="store_true", help="rerun stages even if their inputs did not change")
    r.set_defaults(fn=run)
    a = ap.parse_args()
    a.fn(a)
