"""Stage export: checkpoint -> verified split -> backbone ONNX (4-D) -> onnxsim -> onnx2tf -> int8 TFLite.

Outputs in runs/<name>/export/: info.json (classes, learned front-end parameters, split check), reference.npz,
backbone.onnx, backbone_sim.onnx, tf/ (SavedModel), specs.npz (calibration and evaluation spectrograms),
model.tflite (the int8 backbone) and quantize.json (int8 vs float error, operators).
"""
import json

from wren.config import ROOT
from wren.runner import code, mark_done, need, stage_key, up_to_date

T = ROOT / "wren" / "torchside"


def audio_files(paths):
    files = []
    for p in paths:
        files += sorted(p.rglob("*.wav")) + sorted(p.rglob("*.mp3")) if p.is_dir() else [p]
    return files


def run(r, force=False):
    p = r.p
    out = p.run_dir / "export"
    out.mkdir(parents=True, exist_ok=True)
    prep = p.run_dir / "prepare"
    key = stage_key("export-1", p.checkpoint, [str(a) for a in p.calibration_audio], p.calibration_count, p.eval_count,
                    str(p.reference_audio), p.trains and (prep / "stage.json").read_text(), p.trains and p.hop_seconds,
                    p.trains and code("torchside/specs_from_windows.py"))
    if not force and up_to_date(out, key):
        print("export: up to date"); return out
    print("export")
    if p.trains:  # a training file as the reference clip; calibration from the training windows (below)
        reference = p.reference_audio or json.loads((prep / "files_train.json").read_text())[0]["path"]
    else:
        reference = p.reference_audio or (audio_files(p.calibration_audio) or [None])[0]
    if reference is None: raise SystemExit("no audio files in calibration.audio")
    r.torch("reference", "-m", "wren.torchside.reference", p.checkpoint, reference, out)
    info = json.loads((out / "info.json").read_text())
    if info["classes"] != len(p.classes):
        raise SystemExit(f"the checkpoint has {info['classes']} outputs but project.yaml lists {len(p.classes)} classes")
    r.torch("export_backbone", "-m", "wren.torchside.run", T / "export_backbone.py", p.checkpoint, out / "backbone.onnx", "none", "2d")
    r.torch("onnxsim", "-m", "onnxsim", out / "backbone.onnx", out / "backbone_sim.onnx")
    r.tf("onnx2tf", "-m", "onnx2tf", "-i", out / "backbone_sim.onnx", "-o", out / "tf", "-osd", "-nuo")
    if p.trains:
        r.torch("make_specs", "-m", "wren.torchside.specs_from_windows", prep, p.checkpoint, out / "specs.npz",
                p.calibration_count, p.eval_count, p.training["seed"], p.hop_seconds)
    else:  # WrenNet's script, unchanged: the bird reproduction depends on it
        r.torch("make_specs", "-m", "wren.torchside.run", T / "make_specs.py", p.checkpoint, out / "specs.npz",
                p.calibration_count, p.eval_count, *p.calibration_audio)
    r.tf("quantize", "-m", "wren.tfside.quantize", out / "tf", need(out / "tf" / "backbone_sim_float32.tflite", "float model"),
         out / "specs.npz", p.calibration_count, out / "model.tflite", out / "quantize.json")
    mark_done(out, key)
    return out
