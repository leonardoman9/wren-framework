"""Stage firmware: builds the AudioMoth firmware (normal and diagnostic) from the package and runs the firmware
repository's tests on it (scripts/run_tests.sh: schedule logic, FatFS, flash budget, simulator with fault injection,
QEMU pipeline). Outputs in runs/<name>/firmware/: <name>.bin, <name>_diag.bin, firmware.json. With `firmware: tests:
false` (developers only) the binaries are named <name>_UNTESTED*.bin: constitution IV allows no flashable firmware that
has not passed its tests."""
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
        raise StepFailed("The firmware build tools are not installed: run `python -m wren setup --tflm`")
    key = stage_key("firmware-1", json.loads((package_dir / "stage.json").read_text())["key"], p.firmware_tests)
    if not force and up_to_date(out, key):
        print("firmware: up to date"); return out
    print("firmware")
    # built into a work folder and delivered to firmware/ only after the flash budget and the tests (constitution IV)
    work = p.run_dir / "firmware_build"
    work.mkdir(exist_ok=True)
    for old in out.glob(f"{p.name}*.bin"): old.unlink()
    build = r.run("build", ["sh", fw / "firmware" / "app" / "build_app.sh", p.tflm, package_dir])
    flash = int(re.search(r"flash image: *(\d+) bytes", build).group(1))
    shutil.copy(fw / "firmware" / "app" / "out" / "bird_logger.bin", work / "app.bin")
    r.run("build_diag", ["sh", fw / "firmware" / "app" / "build_app.sh", p.tflm, package_dir], env={"EXTRA_CF": "-DBIRD_DIAG_RECORD"})
    shutil.copy(fw / "firmware" / "app" / "out" / "bird_logger.bin", work / "app_diag.bin")
    if flash > LIMIT: raise StepFailed(f"the firmware needs {flash} bytes of flash, {LIMIT} available (fewer classes?)")
    print(f"  flash {flash} of {LIMIT} bytes")
    if p.firmware_tests:
        # the tests' Python steps need numpy (and ai_edge_litert for QEMU): the TensorFlow environment has both
        path = str(p.tf_python.parent) + os.pathsep + os.environ.get("PATH", "")
        r.run("tests", ["sh", fw / "scripts" / "run_tests.sh", p.tflm, package_dir, "", p.run_dir / "firmware_tests"], env={"PATH": path})
        print("  firmware tests passed")
    stem = p.name if p.firmware_tests else f"{p.name}_UNTESTED"
    shutil.copy(work / "app.bin", out / f"{stem}.bin")
    shutil.copy(work / "app_diag.bin", out / f"{stem}_diag.bin")
    result = {"flash_bytes": flash, "flash_limit": LIMIT, "bin": f"{stem}.bin", "bin_sha256": sha256(out / f"{stem}.bin"),
              "tests": "passed" if p.firmware_tests else "not run: the binaries are marked UNTESTED, do not deploy them"}
    (out / "firmware.json").write_text(json.dumps(result, indent=2) + "\n")
    mark_done(out, key, **result)
    return out
