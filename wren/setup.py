"""python -m wren setup: checks the machine, then installs the two Python environments (torch, TensorFlow) from the
lock files and, with --tflm, fetches and builds TFLM with its pinned ARM compiler (specs/001-reproducible-from-zero,
contracts/cli-setup.md); with --birdnet, fetches BirdNET for `training: birdnet: true` (specs/003-birdnet-distillation,
contracts/setup-birdnet.md). Standard library only: it runs on a bare Python before anything is installed.

Messages are for biologists (constitution I): what is missing and how to fix it, no component names. Versions and
the output of every install step go to runs/_setup/logs/setup.log, and to `--check --verbose`."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from wren import birdnet

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "audiomoth"
TFLM = FW / "third_party" / "tflite-micro"
TFLM_COMMIT = "e0021ca434cd930acbab69768e88d959083a3d4b"
GCC = TFLM / "tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi-gcc"
GCC_VERSION = "14.3.1"
LOG = ROOT / "runs" / "_setup" / "logs" / "setup.log"
MIN_DISK_GB = 6  # environments 2.4 GB + TFLM with its compiler 1.1 GB + pip cache 0.5 GB (measured 2026-10-22), plus 30 %
ENVS = ("torch", "tf")
TOOLS = {"git": "git", "make": "make", "cc": "a C compiler", "clang": "clang", "wget": "wget", "mformat": "mtools",
         "qemu-system-arm": "qemu"}  # clang: the firmware simulator is built with its sanitizers
MAC_TOOLS = {"gmake": "GNU make 4 (macOS's own make is too old for TFLM)"}
INSTALL_HINT = {"linux": "sudo apt install git make gcc clang wget mtools qemu-system-arm",
                "darwin": "xcode-select --install   (git, C compilers), then: brew install make wget mtools qemu"}
DOCKER = "please use Docker instead (see README, 'Using Docker')."


class Stop(Exception):
    """A precondition failed: the message says what to do (exit 2)."""


# ---------------------------------------------------------------- checks

def platform_name(system=None, machine=None, mac_version=None):
    """'linux-x86_64' or 'macos-arm64' (the platforms with lock files); raises Stop on any other."""
    system = system or sys.platform
    machine = (machine or platform.machine()).lower()
    if system.startswith("linux") and machine in ("x86_64", "amd64"):
        return "linux-x86_64"
    if system == "darwin" and machine == "arm64":
        major = int((mac_version if mac_version is not None else platform.mac_ver()[0]).split(".")[0] or 0)
        if major < 14:
            raise Stop(f"This Mac runs macOS {major}; the analysis libraries need macOS 14 or later. Update macOS, or "
                       + DOCKER)
        return "macos-arm64"
    if system == "darwin":
        raise Stop("This Mac has an Intel processor. The analysis libraries no longer support it directly: " + DOCKER)
    raise Stop(f"This computer ({system}, {machine}) is not supported directly: " + DOCKER)


def python_version(exe):
    r = subprocess.run([str(exe), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], capture_output=True,
                       text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def find_python(given=None):
    """The Python 3.11 used to create the environments: --python, this interpreter, or python3.11 on PATH. Symlinks are
    resolved: a venv made through a link to a relocatable Python (uv, pyenv shims) cannot find its standard library."""
    if given:
        found = python_version(given)
        if found != "3.11":
            raise Stop(f"Python 3.11 is needed (found {found or 'nothing'} at {given}). Install it from python.org or with "
                       "your package manager, then run this again.")
        return os.path.realpath(given)
    if sys.version_info[:2] == (3, 11):
        return os.path.realpath(getattr(sys, "_base_executable", None) or sys.executable)  # not a venv's, which may be replaced
    exe = shutil.which("python3.11")
    if exe and python_version(exe) == "3.11":
        return os.path.realpath(exe)
    raise Stop(f"Python 3.11 is needed (found {sys.version_info[0]}.{sys.version_info[1]}). Install it from python.org "
               "or with your package manager, then run this again.")


def check_disk(free_gb=None):
    need = min(MIN_DISK_GB, float(os.environ.get("WREN_MIN_DISK_GB", MIN_DISK_GB)))  # CI runners only (envs/README.md)
    free_gb = shutil.disk_usage(ROOT).free / 1e9 if free_gb is None else free_gb
    if free_gb < need:
        raise Stop(f"About {need:.0f} GB of free disk space is needed; {free_gb:.0f} GB are free.")


def missing_tools(which=shutil.which, system=None):
    tools = {**TOOLS, **(MAC_TOOLS if (system or sys.platform) == "darwin" else {})}
    return [name for exe, name in tools.items() if not which(exe)]


def check_tools(which=shutil.which, system=None):
    system = "darwin" if (system or sys.platform) == "darwin" else "linux"
    missing = missing_tools(which, system)
    if missing:
        raise Stop("Some programs are missing: " + ", ".join(missing) + ".\nInstall them with:\n  "
                   + INSTALL_HINT[system] + "\nthen run this again.")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_file(env, plat):
    """The lock for this environment and platform, checked against its .in file (developer-facing message)."""
    lock = ROOT / "envs" / "lock" / f"{env}-{plat}.txt"
    spec = ROOT / "envs" / f"requirements-{env}.in"
    if not lock.exists():
        raise Stop(f"envs/lock/{lock.name} is missing: regenerate the locks (envs/lock.sh, see envs/README.md)")
    first = lock.read_text().split("\n", 1)[0]
    if first != f"# input-sha256: {sha256(spec)}":
        raise Stop(f"envs/lock/{lock.name} is out of date with envs/{spec.name}: regenerate the locks (envs/lock.sh, "
                   "see envs/README.md)")
    return lock


# ---------------------------------------------------------------- state

def env_ready(env, lock):
    marker = ROOT / f".venv-{env}" / ".wren-lock"
    return not (ROOT / f".venv-{env}").is_symlink() and marker.exists() and marker.read_text().strip() == sha256(lock)


def host_lib():
    return [p for p in TFLM.glob("gen/*_release_with_logs_cmsis_nn_gcc/lib/libtensorflow-microlite.a")
            if "cortex_m" not in str(p)]


def tflm_build_key():
    """What the TFLM build depends on besides its commit: the local patches and the two scripts."""
    files = sorted((FW / "firmware" / "patches").glob("*.patch")) + [FW / "scripts" / "setup_tflm.sh", FW / "scripts" / "build_tflm.sh"]
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()


def tflm_ready():
    libs = [TFLM / f"gen/cortex_m_generic_cortex-m4+fp_{bt}_cmsis_nn_gcc/lib/libtensorflow-microlite.a"
            for bt in ("release", "release_with_logs")]
    if not (all(p.exists() for p in libs) and host_lib() and GCC.exists()):
        return False
    head = subprocess.run(["git", "-C", str(TFLM), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    marker = TFLM / ".wren-build"
    return (head == TFLM_COMMIT and GCC_VERSION in gcc_version() and marker.exists()
            and marker.read_text().strip() == tflm_build_key())


def gcc_version():
    r = subprocess.run([str(GCC), "--version"], capture_output=True, text=True) if GCC.exists() else None
    return r.stdout.split("\n", 1)[0] if r and r.returncode == 0 else ""


# ---------------------------------------------------------------- actions

def step(cmd, what, env=None):
    """Runs one install step, appending its output to the setup log; a failure stops with exit 1."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as log:
        log.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        log.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                           env={**os.environ, **(env or {})})
    if r.returncode != 0:
        print(f"\n{what} failed. The details are in {LOG.relative_to(ROOT)}; please include that file if you ask for "
              "help.", file=sys.stderr)
        sys.exit(1)


def install_env(env, lock, python):
    venv = ROOT / f".venv-{env}"
    if venv.is_symlink():
        venv.unlink()  # the link only, never what it points to
    elif venv.exists():
        shutil.rmtree(venv)  # stale: rebuilt from the lock, so nothing outside it lingers
    what = "Installing the analysis tools"
    step([python, "-m", "venv", venv], what)
    step([venv / "bin" / "python", "-m", "pip", "install", "--quiet", "--require-hashes", "--no-deps", "-r", lock], what)
    (venv / ".wren-lock").write_text(sha256(lock) + "\n")


def install_tflm():
    what = "Preparing the firmware build tools"
    if (TFLM / ".git").exists():  # a rebuild after the patches changed: back to the pristine commit first
        step(["git", "-C", TFLM, "checkout", "--", "."], what)
    step(["sh", FW / "scripts" / "setup_tflm.sh"], what)
    step(["sh", FW / "scripts" / "build_tflm.sh"], what, env={"HOST": "1"})
    if GCC_VERSION not in gcc_version():
        print(f"\n{what} failed: an unexpected compiler was installed. Details in {LOG.relative_to(ROOT)}.",
              file=sys.stderr)
        sys.exit(1)
    (TFLM / ".wren-build").write_text(tflm_build_key() + "\n")


def install_birdnet():
    """BirdNET's model and species list from Zenodo, checked against their pinned hashes; the marker is written last, so a
    failed or interrupted download leaves BirdNET 'missing', never half-installed."""
    what = "Downloading BirdNET"
    d = birdnet.DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / birdnet.MARKER).unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=d) as tmp:
            zf = Path(tmp) / "birdnet.zip"
            with urllib.request.urlopen(birdnet.ZIP_URL, timeout=60) as r, open(zf, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if birdnet.sha256(zf) != birdnet.ZIP_SHA256:
                raise Stop(f"{what} failed: the file that arrived is not the expected one. Please try again later.")
            with zipfile.ZipFile(zf) as z:
                for name in birdnet.FILES:
                    (d / name).parent.mkdir(parents=True, exist_ok=True)
                    (d / name).write_bytes(z.read(name))
        for name, expected in birdnet.FILES.items():
            if birdnet.sha256(d / name) != expected:
                raise Stop(f"{what} failed: an unpacked file is not the expected one. Please try again later.")
    except OSError as e:
        for name in birdnet.FILES: (d / name).unlink(missing_ok=True)
        raise Stop(f"{what} failed ({e}). Check the internet connection and run this again.")
    except Stop:
        for name in birdnet.FILES: (d / name).unlink(missing_ok=True)
        raise
    (d / birdnet.MARKER).write_text(json.dumps(birdnet.FILES))


# ---------------------------------------------------------------- entry points

def versions(plat):
    """Developer details for --verbose and the log."""
    out = [f"platform {plat}"]
    for env, pkg in (("torch", "torch"), ("tf", "tensorflow")):
        py = ROOT / f".venv-{env}" / "bin" / "python"
        r = subprocess.run([str(py), "-c", f"import sys, importlib.metadata as m; print(sys.version.split()[0], m.version('{pkg}'))"],
                           capture_output=True, text=True) if py.exists() else None
        out.append(f".venv-{env}: " + (r.stdout.strip().replace(" ", f", {pkg} ") if r and r.returncode == 0 else "missing"))
    head = subprocess.run(["git", "-C", str(TFLM), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    out.append(f"TFLM: {head.stdout.strip() or 'missing'}; arm-none-eabi-gcc: {gcc_version() or 'missing'}")
    return out


def report(verbose=False):
    """Reports every item in plain words, changes nothing; 0 if everything is ready."""
    rows, ok = [], True

    def row(name, fine, fix=""):
        nonlocal ok
        ok &= fine
        rows.append(f"{name}: " + ("ready" if fine else f"missing — {fix}"))

    try:
        plat = platform_name()
        row("This computer", True)
    except Stop as e:
        print(e)
        return 2
    try:
        find_python()
        row("Python 3.11", True)
    except Stop:
        row("Python 3.11", False, "install Python 3.11")
    missing = missing_tools()
    row("System programs", not missing, "install " + ", ".join(missing) + " (see README, 'Setup')")
    try:
        ready = all(env_ready(env, lock_file(env, plat)) for env in ENVS)
    except Stop:
        ready = False
    row("Analysis tools", ready, "run python -m wren setup")
    row("Firmware build tools", tflm_ready(), "run python -m wren setup --tflm")
    rows.append("Learning from BirdNET (optional): " + ("ready" if birdnet.ready() else "not installed — only needed for "
                                                         "training: birdnet: true; run python -m wren setup --birdnet"))
    print("\n".join(rows + (["", *versions(plat)] if verbose else [])))
    return 0 if ok else 1


def main(tflm=False, check=False, verbose=False, python=None, birdnet_too=False):
    if check:
        return report(verbose)
    try:
        plat = platform_name()
        py = find_python(python)
        locks = {env: lock_file(env, plat) for env in ENVS}
        todo = [env for env in ENVS if not env_ready(env, locks[env])]
        build = tflm and not tflm_ready()
        if todo or build:
            check_disk()
        check_tools()
    except Stop as e:
        print(e, file=sys.stderr)
        return 2
    if todo:
        print("Installing the analysis tools (a few minutes the first time)…", flush=True)
        for env in todo:
            install_env(env, locks[env], py)
    if build:
        print("Preparing the firmware build tools (about 10 minutes the first time)…", flush=True)
        install_tflm()
    if birdnet_too:
        if birdnet.ready():
            print("BirdNET: ready")
        else:
            print("Downloading BirdNET (about 77 MB, once)…", flush=True)
            try:
                install_birdnet()
            except Stop as e:
                print(e, file=sys.stderr)
                return 1
        print(f"BirdNET v2.4 ({birdnet.CITATION}) is licensed {birdnet.LICENCE}: non-commercial use. Models trained with it "
              "as a teacher may fall under the same condition.")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as log:
        log.write("\n".join(versions(plat)) + "\n")
    print("Ready. Next: python -m wren run your-project.yaml" if tflm or tflm_ready()
          else "Ready for analysis. For the firmware, also run: python -m wren setup --tflm")
    return 0
