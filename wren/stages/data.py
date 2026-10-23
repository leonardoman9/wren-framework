"""Stages prepare, train, evaluate (training projects): audio folders -> labelled windows -> checkpoint -> test metrics.
Options: README.md (project.yaml). Outputs: runs/<name>/prepare/, train/ (best.pt), evaluate/."""
import json
import os

from wren import birdnet
from wren.config import ROOT
from wren.runner import StepFailed, code, listing, mark_done, model_code, stage_key, up_to_date


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


def train_key(prep_key, spec, *extra):
    """The prepared windows, the settings, the training code, the starting checkpoint (by content), the model code
    and, when learning from BirdNET, BirdNET's results and the target code (extra)."""
    return stage_key("train-1", prep_key, spec, code("torchside/train.py", "torchside/model.py", "torchside/metrics.py"),
                     ROOT / "wrennet" / "checkpoints" / "birds.pt", model_code(), *extra)


def source_hashes(prep_dir, cache):
    """sha256 of every training file's source audio, by path; reused while a file's size and modification time are
    unchanged, so an unchanged dataset is not read again."""
    old = json.loads(cache.read_text()) if cache.exists() else {}
    new = {}
    for f in json.loads((prep_dir / "files_train.json").read_text()):
        st = os.stat(f["path"])
        hit = old.get(f["path"])
        new[f["path"]] = hit if hit and hit[:2] == [st.st_size, int(st.st_mtime)] else [st.st_size, int(st.st_mtime), birdnet.sha256(f["path"])]
    cache.write_text(json.dumps(new))
    return {path: v[2] for path, v in new.items()}


def ask_birdnet(r, prep_dir):
    """BirdNET's opinion of the training audio (specs/003-birdnet-distillation, research R8), kept in train/birdnet/ under
    a key made of file contents, the grid, the species and BirdNET's files: no paths or times, so a copied run folder is
    reused on a computer without BirdNET while the training recordings are the same. --force does not ask again: the key
    already follows everything the results depend on."""
    p = r.p
    out = p.run_dir / "train" / "birdnet"
    out.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes(prep_dir, out / "hashes.json")
    species = p.training["birdnet_species"]
    hop = p.data["windows"]["hop_seconds"]
    key = stage_key("birdnet-1", sorted(hashes.values()), hop, species, birdnet.FILES,
                    code("torchside/birdnet_windows.py", "tfside/birdnet.py"))
    if up_to_date(out, key) and (out / "teacher.npz").exists():
        print("  BirdNET's results: up to date"); return out, key
    if not birdnet.ready():
        raise StepFailed("BirdNET is needed for the new training audio: run python -m wren setup --birdnet")
    folders = [f for f in p.class_folders if f in species]  # class order, background left out
    (out / "hashes_by_path.json").write_text(json.dumps(hashes))
    (out / "indices.json").write_text(json.dumps([species[f]["index"] for f in folders]))
    r.torch("birdnet_windows", "-m", "wren.torchside.birdnet_windows", prep_dir, hop, out / "hashes_by_path.json", out)
    r.tf("birdnet", ROOT / "wren" / "tfside" / "birdnet.py", birdnet.DIR, out, out / "indices.json", out)
    (out / "audio48.raw").unlink()
    (out / "mapping.json").write_text(json.dumps({"birdnet_sha256": birdnet.FILES, "species": species}, indent=2) + "\n")
    mark_done(out, key)
    return out, key


def train(r, prep_dir, prep_key, force=False):
    p = r.p
    out = p.run_dir / "train"
    out.mkdir(parents=True, exist_ok=True)
    spec = {"classes": p.classes, "negative": p.negative_index, **p.training,
            "wrennet_checkpoint": str(ROOT / "wrennet" / "checkpoints" / "birds.pt")}
    extra = []
    if p.training.get("birdnet"):
        teacher_dir, teacher_key = ask_birdnet(r, prep_dir)
        spec["birdnet_teacher"] = str(teacher_dir / "teacher.npz")
        spec["birdnet_names"] = [p.training["birdnet_species"][f]["name"] for f in p.class_folders if f in p.training["birdnet_species"]]
        extra = [teacher_key, code("torchside/birdnet_target.py")]
    key = train_key(prep_key, spec, *extra)
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
