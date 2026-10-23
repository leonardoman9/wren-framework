"""Stage package: the model package the firmware builds from (layout: audiomoth/firmware/bird/model/README.md).

runs/<name>/package/: model.tflite, model_config.json, generated/ (bird_frontend_tables.h, bird_tail_weights_i8.h,
tail_dims.h, model_config.h), test/ (pipe_vec.cc, specs_one.npz: one window and its expected logits for the QEMU
test). Every file is made by the firmware repository's own generators. The device pipeline (C front-end + int8
backbone + C tail, built for the host) is checked against PyTorch on test windows: package.json.
"""
import json
import re
import shutil

from wren.runner import code, content, mark_done, need, sha256, stage_key, tree_hash, up_to_date


def run(r, export_dir, force=False):
    r.pinned = True  # byte-identical on every x86 computer (runner.PINNED)
    p = r.p
    fw = need(p.firmware_repo, "firmware folder (audiomoth/)")
    bird = fw / "firmware" / "bird"
    out = p.run_dir / "package"
    work = p.run_dir / "package_work"
    prep_state = p.run_dir / "prepare" / "stage.json"
    key = stage_key("package-3", p.trains, p.trains and prep_state.exists() and prep_state.read_text(),
                    p.trains and code("torchside/windows_ref.py", "tfside/device_eval.py"), export_dir / "model.tflite", export_dir / "reference.npz", p.checkpoint, p.classes,
                    p.negative, p.window_start, p.window_minutes, p.hop_seconds, p.confidence,
                    content(*p.test_audio), p.test_windows, tree_hash(fw), code("stages/package.py", "runner.py"))
    if not force and up_to_date(out, key):
        print("package: up to date"); return out
    print("package")
    shutil.rmtree(out, ignore_errors=True); shutil.rmtree(work, ignore_errors=True)
    gen, test = out / "generated", out / "test"
    for d in (gen, test, work): d.mkdir(parents=True)
    info = json.loads((export_dir / "info.json").read_text())

    # learned filterbank -> constant tables of the C front-end
    r.run("frontend_tables_build", ["cc", "-O2", fw / "tools" / "gen_frontend_tables.c", "-lm", "-o", work / "gen_frontend_tables"])
    tables = r.run("frontend_tables", [work / "gen_frontend_tables", repr(info["breakpoint"]), repr(info["transition_width"]),
                                       repr(info["f_min"]), repr(info["f_max"])])
    (gen / "bird_frontend_tables.h").write_text(tables)

    # GRU, projection, attention and classifier weights (int8 per row) and the frame count
    r.torch("tail_weights", bird / "gen_bird_tail.py", p.checkpoint, export_dir / "reference.npz", work / "tail")
    for f in ("bird_tail_weights_i8.h", "tail_dims.h"): shutil.copy(work / "tail" / f, gen / f)

    # backbone, classes and device settings
    shutil.copy(export_dir / "model.tflite", out / "model.tflite")
    r.run("model_config", ["python3", bird / "gen_model_config.py", out, "--names", ",".join(p.classes), "--negative", p.negative,
                           "--window-start", p.window_start, "--window-minutes", p.window_minutes,
                           "--hop-seconds", p.hop_seconds, "--confidence", p.confidence])

    # the device pipeline on the host vs PyTorch, and one window with its expected logits for the QEMU test
    for name, src in (("libfe", "bird_frontend.c"), ("libtail", "bird_tail.c")):
        r.run(f"{name}_build", ["cc", "-O2", "-shared", "-fPIC", f"-I{bird}", f"-I{gen}", bird / src, "-lm", "-o", work / f"{name}.so"])
    if p.trains:  # labelled test windows of the trained model
        r.torch("test_reference", "-m", "wren.torchside.windows_ref", p.run_dir / "prepare", "test", p.checkpoint,
                work / "e2e_ref.npz", max(p.test_windows, 2000), p.training["seed"])
    else:
        r.torch("test_reference", bird / "e2e_make_ref.py", p.checkpoint, work / "e2e_ref.npz", p.test_windows, *p.test_audio)
    check = r.tf("pipeline_check", bird / "e2e_check.py", work / "e2e_ref.npz", out / "model.tflite", work / "libfe.so", work / "libtail.so")
    m = re.search(r"windows=(\d+) \| argmax agree=(\d+)/(\d+) \| logits max\|d\|=([\d.]+)", check)
    if not m: raise SystemExit("The check of the device version gave an unexpected result; details in logs/pipeline_check.log")
    windows, agree = int(m.group(1)), int(m.group(2))
    r.tf("test_vectors", bird / "gen_pipeline_vec.py", work / "e2e_ref.npz", 0, out / "model.tflite", work / "libfe.so",
         work / "libtail.so", test / "pipe_vec.cc")
    r.tf("test_spec", "-c", "import numpy as np, sys; np.savez_compressed(sys.argv[2], eval=np.load(sys.argv[1])['eval'][:1])",
         export_dir / "specs.npz", test / "specs_one.npz")

    result = {"windows": windows, "class_agreement": agree, "logits_max_diff": float(m.group(4)),
              "model_sha256": sha256(out / "model.tflite"), "checkpoint_sha256": sha256(p.checkpoint)}
    if p.trains:  # what int8 costs on the test windows, measured with the firmware's own C code
        r.tf("device_eval", "-m", "wren.tfside.device_eval", work / "e2e_ref.npz", out / "model.tflite", work / "libfe.so",
             work / "libtail.so", len(p.classes), work / "device_eval.json")
        d = json.loads((work / "device_eval.json").read_text())
        result["device_eval"] = d
        print(f"  test windows: macro-F1 float {d['float_macro_f1']:.3f} -> device int8 {d['int8_macro_f1']:.3f}")
        if d["float_macro_f1"] - d["int8_macro_f1"] > 0.05:
            raise SystemExit(f"The device version loses {d['float_macro_f1'] - d['int8_macro_f1']:.3f} macro-F1 on the test "
                             "windows compared with the original model (the limit is 0.05)")
    (out / "package.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"  device pipeline vs PyTorch: same class in {agree}/{windows} windows")
    if not p.trains and agree < 0.9 * windows:
        raise SystemExit(f"The device version disagrees with the original model on {windows - agree} of {windows} test windows")
    mark_done(out, key, **result)
    return out
