"""Stage report: runs/<name>/report.md, what was made and how well it matches the checkpoint."""
import json
from datetime import datetime, timezone


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def run(r):
    p = r.p
    d = p.run_dir
    info, quant = load(d / "export" / "info.json"), load(d / "export" / "quantize.json")
    pkg, fw = load(d / "package" / "package.json"), load(d / "firmware" / "firmware.json")
    rows = [
        ("checkpoint", f"`{p.checkpoint}` (sha256 `{pkg.get('checkpoint_sha256', '?')[:16]}…`)"),
        ("classes", f"{len(p.classes)}: " + ", ".join(p.classes)),
        ("front-end (learned)", f"breakpoint {info.get('breakpoint', 0):.2f} Hz, transition width {info.get('transition_width', 0):.2f}, "
                                f"{info.get('f_min', 0):.0f}-{info.get('f_max', 0):.0f} Hz"),
        ("split check", f"front-end + core vs whole model: max \\|d\\| {info.get('split_max_diff', float('nan')):.2e}; "
                        f"ONNX vs PyTorch {info.get('onnx_max_diff', float('nan')):.2e}"),
        ("int8 backbone", f"`package/model.tflite`, {quant.get('bytes', '?')} bytes, sha256 `{pkg.get('model_sha256', '?')[:16]}…`, "
                          f"{sum(quant.get('operators', {}).values())} operators"),
        ("int8 vs float", f"relative RMS error {quant.get('int8_vs_float_rel_rms', float('nan')):.3f} on {quant.get('eval', '?')} spectrograms "
                          f"(calibrated on {quant.get('calibration', '?')})"),
        ("device pipeline vs PyTorch", f"same class in {pkg.get('class_agreement', '?')}/{pkg.get('windows', '?')} windows, "
                                       f"max \\|Δlogit\\| {pkg.get('logits_max_diff', float('nan')):.3f}"),
        ("device settings", f"window {p.window_start} for {p.window_minutes} min, hop {p.hop_seconds} s, confidence {p.confidence} %"),
        ("firmware", f"`firmware/{p.name}.bin`, {fw.get('flash_bytes', '?')} of {fw.get('flash_limit', '?')} bytes of flash; "
                     f"diagnostic build `firmware/{p.name}_diag.bin`" if fw else "not built"),
        ("firmware tests", fw.get("tests", "not run")),
    ]
    text = f"# {p.name}\n\nMade {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by wren-framework.\n\n| | |\n|---|---|\n"
    text += "".join(f"| {k} | {v} |\n" for k, v in rows)
    text += ("\nFlash the `.bin` with the AudioMoth Flash App; the card needs `CONFIG.TXT` (see the firmware README). "
             "Logs of every step: `logs/`.\n")
    (d / "report.md").write_text(text)
    print(f"report: {d / 'report.md'}")
