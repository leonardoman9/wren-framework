"""Stage report: runs/<name>/report.md, what was made and how well it matches the checkpoint."""
import json
import os
import subprocess
from datetime import datetime, timezone

from wren import setup
from wren.config import ROOT
from wren.runner import sha256, tree_hash


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def provenance():
    """How this run was made (FR-018, constitution VI): only the Docker reference path is byte-reproducible."""
    docker = os.environ.get("WREN_PATH") == "docker"
    try:
        plat = setup.platform_name()
    except setup.Stop:
        plat = "unsupported"
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    locks = {f.name: sha256(f) for env in setup.ENVS for f in [ROOT / "envs" / "lock" / f"{env}-{plat}.txt"] if f.exists()}
    return {"path": "docker" if docker else "native", "image": os.environ.get("WREN_IMAGE_ID") if docker else None,
            "platform": plat, "locks": locks,
            "sources": {"commit": git("rev-parse", "HEAD") or "unknown", "dirty": bool(git("status", "--porcelain")),
                        "wrennet": tree_hash(ROOT / "wrennet"), "audiomoth": tree_hash(ROOT / "audiomoth")},
            "byte_reproducible": docker}


def summary(p, pkg, fw, prov):
    """The plain-language part of the report (constitution I): what was made, does it fit, did every check pass."""
    tested = fw.get("tests") == "passed"
    agree = pkg.get("class_agreement") == pkg.get("windows") if pkg else False
    lines = [
        f"- **Firmware:** `firmware/{fw.get('bin', p.name + '.bin')}`, for the AudioMoth Flash App" + ("" if tested else
            " (**not tested: do not deploy it**)") if fw else "- **Firmware:** not built",
        f"- **Classes:** {len(p.classes)}: " + ", ".join(p.classes),
        f"- **Fits on the device:** {'yes' if fw and fw.get('flash_bytes', 0) <= fw.get('flash_limit', 0) else 'no'}"
        + (f" ({fw['flash_bytes']:,} of {fw['flash_limit']:,} bytes)" if fw else ""),
        f"- **Device version gives the same answers as the original model:** "
        + (f"yes, on all {pkg['windows']} test windows" if agree else
           f"on {pkg.get('class_agreement', '?')} of {pkg.get('windows', '?')} test windows"),
        f"- **Firmware tests on a simulated device:** {'passed' if tested else 'not run'}",
        "- **Made with:** " + ("the Docker reference path; the same inputs give byte-identical files on any computer"
                               if prov["byte_reproducible"] else
                               "a native install; verified by the checks above, but the bytes may differ from Docker's"),
    ]
    return "## Summary\n\n" + "\n".join(lines) + "\n\n"


def run(r):
    p = r.p
    d = p.run_dir
    info, quant = load(d / "export" / "info.json"), load(d / "export" / "quantize.json")
    pkg, fw = load(d / "package" / "package.json"), load(d / "firmware" / "firmware.json")
    prov = provenance()
    (d / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
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
                          f"(calibrated on {quant.get('calibration', '?')}"
                          + (f" of {quant['calibration_requested']} requested: a small dataset" if quant.get("calibration_requested", 0) >
                             quant.get("calibration", 0) else "") + ")"),
        ("device pipeline vs PyTorch", f"same class in {pkg.get('class_agreement', '?')}/{pkg.get('windows', '?')} windows, "
                                       f"max \\|Δlogit\\| {pkg.get('logits_max_diff', float('nan')):.3f}"),
        ("device settings", f"window {p.window_start} for {p.window_minutes} min, hop {p.hop_seconds} s, confidence {p.confidence} %"),
        ("firmware", f"`firmware/{fw.get('bin', p.name + '.bin')}`, {fw.get('flash_bytes', '?')} of {fw.get('flash_limit', '?')} "
                     f"bytes of flash; diagnostic build `firmware/{fw.get('bin', p.name + '.bin')[:-4]}_diag.bin`" if fw else "not built"),
        ("firmware tests", fw.get("tests", "not run")),
    ]
    rows.append(("made with", "Docker reference path: byte-reproducible" if prov["byte_reproducible"] else
                 "native install: verified by parity checks, not byte-reproducible"))
    text = f"# {p.name}\n\nMade {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by wren-framework.\n\n" + summary(p, pkg, fw, prov)
    text += "## Technical details\n\n"
    if p.trains:
        prep, tr, ev = load(d / "prepare" / "prepare.json"), load(d / "train" / "train.json"), load(d / "evaluate" / "evaluate.json")
        dev = pkg.get("device_eval", {})
        text += "### Data and training\n\n| | |\n|---|---|\n"
        drows = [
            ("data", ", ".join(f"{k}: {v['files']} files, {v['windows']} windows" for k, v in prep.get("splits", {}).items())),
            ("training", f"init {tr.get('init')}, best epoch {tr.get('best_epoch')} of {tr.get('epochs_run')}, "
                         f"validation macro-F1 {tr.get('best', {}).get('val_macro_f1', float('nan')):.3f}, {tr.get('device')}, "
                         f"{tr.get('seconds', 0) / 60:.1f} min"),
            ("test (call windows, float)", f"macro-F1 {ev.get('call_windows', {}).get('macro_f1', float('nan')):.3f}, accuracy "
                                           f"{ev.get('call_windows', {}).get('accuracy', float('nan')):.3f} on {ev.get('test_windows')} windows "
                                           f"(chance {ev.get('chance_macro_f1', float('nan')):.3f})"),
            ("test on the device (int8, C)", f"macro-F1 {dev.get('int8_macro_f1', float('nan')):.3f} (float {dev.get('float_macro_f1', float('nan')):.3f}) "
                                             f"on {dev.get('windows')} windows"),
            ("device decision per file", f"suggested confidence {ev.get('files', {}).get('suggested_confidence_pct')} %; at the configured "
                                         f"{p.confidence} %: file macro-F1 {ev.get('files', {}).get('at_configured', {}).get('macro_f1')}, "
                                         f"false reports/hour on background {ev.get('files', {}).get('at_configured', {}).get('false_reports_per_hour')}"),
        ]
        bands = prep.get("class_bands_hz", {})
        drows.insert(1, ("windows chosen within", ", ".join(f"{c.replace('|', chr(92) + '|')} {lo}-{hi} Hz" for c, (lo, hi) in bands.items()) +
                         ("; " if bands else "") + "other classes: the whole range"))
        text += "".join(f"| {k} | {v} |\n" for k, v in drows)
        if prep.get("warnings"): text += "\nWarnings from prepare:\n" + "".join(f"- {w}\n" for w in prep["warnings"][:10])
        per = ev.get("call_windows", {}).get("per_class", {})
        if per:
            text += "\nPer class (test call windows): " + ", ".join(f"{c} F1 {v['f1']:.2f}" for c, v in per.items()) + "\n"
        text += "\n### Conversion and firmware\n\n"
    text += "| | |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in rows)
    text += ("\nFlash the `.bin` with the AudioMoth Flash App; the card needs `CONFIG.TXT` (see the firmware README). "
             "Logs of every step: `logs/`.\n")
    (d / "report.md").write_text(text)
    print(f"report: {d / 'report.md'}")
