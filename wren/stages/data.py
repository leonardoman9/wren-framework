"""Stages prepare, train, evaluate (training projects): audio folders -> labelled windows -> checkpoint -> test metrics.
See docs/PLAN_STEP3.md. Outputs: runs/<name>/prepare/, train/ (best.pt), evaluate/."""
import json

from wren.config import ROOT
from wren.runner import code, listing, mark_done, stage_key, submodule_commit, up_to_date


def _spec(dir_, name, spec):
    f = dir_ / f"{name}_spec.json"
    f.write_text(json.dumps(spec, indent=2) + "\n")
    return f


def prepare(r, force=False):
    p = r.p
    out = p.run_dir / "prepare"
    out.mkdir(parents=True, exist_ok=True)
    spec = {"classes": p.class_folders, "negative": p.negative_index, "sources": p.data["sources"], "split": p.data["split"],
            "group": p.data["group"], "windows": p.data["windows"]}
    key = stage_key("prepare-1", spec, listing(*p.data["sources"].values()), code("torchside/prepare.py"))
    if not force and up_to_date(out, key):
        print("prepare: up to date"); return out, key
    print("prepare")
    r.torch("prepare", "-m", "wren.torchside.prepare", _spec(out, "prepare", spec), out)
    s = json.loads((out / "prepare.json").read_text())
    print("  " + ", ".join(f"{k}: {v['files']} files / {v['windows']} windows" for k, v in s["splits"].items()))
    for w in s["warnings"][:8]: print("  warning:", w)
    if len(s["warnings"]) > 8: print(f"  ... {len(s['warnings']) - 8} more warnings in {out / 'prepare.json'}")
    bad = {c: sum(v[sp]["unreadable"] for sp in v) for c, v in s["per_class"].items() if sum(v[sp]["unreadable"] for sp in v)}
    if bad: print(f"  unreadable files skipped, per class: {bad}")
    mark_done(out, key)
    return out, key


def train(r, prep_dir, prep_key, force=False):
    p = r.p
    out = p.run_dir / "train"
    out.mkdir(parents=True, exist_ok=True)
    spec = {"classes": p.classes, "negative": p.negative_index, **p.training,
            "wrennet_checkpoint": str(ROOT / "third_party" / "WrenNet" / "best_distillation_model.pt")}
    key = stage_key("train-1", prep_key, spec, code("torchside/train.py", "torchside/model.py", "torchside/metrics.py"),
                    ROOT / "third_party" / "WrenNet" / "best_distillation_model.pt", submodule_commit("WrenNet"))
    if not force and up_to_date(out, key) and (out / "best.pt").exists():
        print("train: up to date"); return out, key
    print(f"train ({p.training['init']}, up to {p.training['epochs']} epochs)")
    r.torch("train", "-m", "wren.torchside.train", _spec(out, "train", spec), prep_dir, out)
    t = json.loads((out / "train.json").read_text())
    print(f"  best epoch {t['best_epoch']} of {t['epochs_run']}: validation macro-F1 {t['best']['val_macro_f1']:.3f}, "
          f"accuracy {t['best']['val_accuracy']:.3f} ({t['device']}, {t['seconds'] / 60:.1f} min)")
    mark_done(out, key)
    return out, key


def evaluate(r, prep_dir, train_key, force=False):
    p = r.p
    out = p.run_dir / "evaluate"
    out.mkdir(parents=True, exist_ok=True)
    spec = {"classes": p.classes, "negative": p.negative_index, "device_hop_seconds": p.hop_seconds, "device_confidence": p.confidence}
    key = stage_key("evaluate-1", train_key, spec, code("torchside/evaluate.py", "torchside/metrics.py"))
    if not force and up_to_date(out, key):
        print("evaluate: up to date"); return out
    print("evaluate")
    r.torch("evaluate", "-m", "wren.torchside.evaluate", _spec(out, "evaluate", spec), prep_dir, p.checkpoint, out)
    e = json.loads((out / "evaluate.json").read_text())
    print(f"  test: macro-F1 {e['call_windows']['macro_f1']:.3f}, accuracy {e['call_windows']['accuracy']:.3f} "
          f"({e['test_windows']} windows; chance {e['chance_macro_f1']:.3f}); suggested confidence {e['files']['suggested_confidence_pct']} % "
          f"(from validation)")
    mark_done(out, key)
    return out
