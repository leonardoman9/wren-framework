"""Stage firmware: builds the AudioMoth firmware (normal and diagnostic) from the package and runs the firmware
repository's tests on it (scripts/run_tests.sh: schedule logic, FatFS, flash budget, simulator with fault injection,
QEMU pipeline). Outputs in runs/<name>/firmware/: <name>.bin, <name>_diag.bin, firmware.json."""
import json
import os
import re
import shutil

from wren.runner import StepFailed, mark_done, sha256, stage_key, up_to_date

LIMIT = 245760  # AudioMoth application flash (0x4000-0x40000)


def run(r, package_dir, force=False):
    p = r.p
    fw = p.firmware_repo
    out = p.run_dir / "firmware"
    out.mkdir(parents=True, exist_ok=True)
    lib = p.tflm / "gen" / "cortex_m_generic_cortex-m4+fp_release_cmsis_nn_gcc" / "lib" / "libtensorflow-microlite.a"
    if not lib.exists():
        raise StepFailed(f"TFLM is not built in {p.tflm}: run `python -m wren setup --tflm` (or, in {fw}: "
                         "scripts/setup_tflm.sh && HOST=1 scripts/build_tflm.sh)")
    key = stage_key("firmware-1", json.loads((package_dir / "stage.json").read_text())["key"], p.firmware_tests)
    if not force and up_to_date(out, key):
        print("firmware: up to date"); return out
    print("firmware")
    build = r.run("build", ["sh", fw / "firmware" / "app" / "build_app.sh", p.tflm, package_dir])
    flash = int(re.search(r"flash image: *(\d+) bytes", build).group(1))
    shutil.copy(fw / "firmware" / "app" / "out" / "bird_logger.bin", out / f"{p.name}.bin")
    r.run("build_diag", ["sh", fw / "firmware" / "app" / "build_app.sh", p.tflm, package_dir], env={"EXTRA_CF": "-DBIRD_DIAG_RECORD"})
    shutil.copy(fw / "firmware" / "app" / "out" / "bird_logger.bin", out / f"{p.name}_diag.bin")
    result = {"flash_bytes": flash, "flash_limit": LIMIT, "bin_sha256": sha256(out / f"{p.name}.bin"), "tests": "not run"}
    if flash > LIMIT: raise StepFailed(f"the firmware needs {flash} bytes of flash, {LIMIT} available (fewer classes?)")
    print(f"  flash {flash} of {LIMIT} bytes")
    if p.firmware_tests:
        # the tests' Python steps need numpy (and ai_edge_litert for QEMU): the TensorFlow environment has both
        path = str(p.tf_python.parent) + os.pathsep + os.environ.get("PATH", "")
        r.run("tests", ["sh", fw / "scripts" / "run_tests.sh", p.tflm, package_dir, "", p.run_dir / "firmware_tests"], env={"PATH": path})
        result["tests"] = "passed"
        print("  firmware tests passed")
    (out / "firmware.json").write_text(json.dumps(result, indent=2) + "\n")
    mark_done(out, key, **result)
    return out
