"""Stage export: checkpoint -> verified split -> backbone ONNX (4-D) -> onnxsim -> onnx2tf -> int8 TFLite.

Outputs in runs/<name>/export/: info.json (classes, learned front-end parameters, split check), reference.npz,
backbone.onnx, backbone_sim.onnx, tf/ (SavedModel), specs.npz (calibration and evaluation spectrograms),
model.tflite (the int8 backbone) and quantize.json (int8 vs float error, operators).
"""
import json

from wren.config import ROOT
from wren.runner import mark_done, need, stage_key, up_to_date

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
    key = stage_key("export-1", p.checkpoint, [str(a) for a in p.calibration_audio], p.calibration_count, p.eval_count,
                    str(p.reference_audio))
    if not force and up_to_date(out, key):
        print("export: up to date"); return out
    print("export")
    reference = p.reference_audio or (audio_files(p.calibration_audio) or [None])[0]
    if reference is None: raise SystemExit("no audio files in calibration.audio")
    r.torch("reference", "-m", "wren.torchside.reference", p.checkpoint, reference, out)
    info = json.loads((out / "info.json").read_text())
    if info["classes"] != len(p.classes):
        raise SystemExit(f"the checkpoint has {info['classes']} outputs but project.yaml lists {len(p.classes)} classes")
    r.torch("export_backbone", "-m", "wren.torchside.run", T / "export_backbone.py", p.checkpoint, out / "backbone.onnx", "none", "2d")
    r.torch("onnxsim", "-m", "onnxsim", out / "backbone.onnx", out / "backbone_sim.onnx")
    r.tf("onnx2tf", "-m", "onnx2tf", "-i", out / "backbone_sim.onnx", "-o", out / "tf", "-osd", "-nuo")
    r.torch("make_specs", "-m", "wren.torchside.run", T / "make_specs.py", p.checkpoint, out / "specs.npz",
            p.calibration_count, p.eval_count, *p.calibration_audio)
    r.tf("quantize", "-m", "wren.tfside.quantize", out / "tf", need(out / "tf" / "backbone_sim_float32.tflite", "float model"),
         out / "specs.npz", p.calibration_count, out / "model.tflite", out / "quantize.json")
    mark_done(out, key)
    return out
