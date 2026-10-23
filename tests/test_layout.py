"""The repository holds only code that is used (specs/002-clean-wrennet-audiomoth): wrennet/ is the model package and the
bird checkpoint, no file names the old WrenNet paths, and the audiomoth/ files removed as unused stay removed.
Walks the folders instead of asking git, so that it also runs in the Docker image (no .git there).
Usage: .venv-torch/bin/python -m pytest tests/test_layout.py"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", "third_party", "out", ".pytest_cache"}  # caches, build output, fetched code
SEARCHED = ["wren", "wrennet", "audiomoth", "tests", "tools", "docs", "examples", "docker", "envs", ".github", "README.md"]
STALE = ("WRENNET_ROOT", "birds_distillation_edge", "best_distillation_model.pt")
REMOVED = ["audiomoth/docs/datasheets", "audiomoth/docs/images/image.png", "audiomoth/firmware/snippets",
           "audiomoth/tools/bird_frontend_v1_reference.c.txt", "audiomoth/firmware/bird/host_test.c",
           "audiomoth/firmware/bird/map_sizes.py", "audiomoth/firmware/bird/sweep_tail.py", "audiomoth/tests/sim/check_wav.py"]


def files(top):
    top = ROOT / top
    if top.is_file(): return [top]
    found = []
    for d, subdirs, names in os.walk(top):
        subdirs[:] = [s for s in subdirs if s not in SKIP_DIRS and not s.startswith("out_")]
        found += [Path(d) / n for n in names if not n.endswith(".pyc") and n != ".DS_Store"]
    return found


def test_wrennet_is_the_model_package():
    have = sorted(str(f.relative_to(ROOT / "wrennet")) for f in files("wrennet"))
    assert have == ["README.md", "__init__.py", "checkpoints/birds.pt",
                    "models/__init__.py", "models/core.py", "models/legacy_models.py", "models/legacy_modules.py",
                    "spectral/__init__.py", "spectral/differentiable_spec.py"]


def test_no_stale_wrennet_names():
    stale = []
    for f in (f for top in SEARCHED for f in files(top)):
        if f == Path(__file__).resolve(): continue
        try: text = f.read_text()
        except (UnicodeDecodeError, OSError): continue  # binary
        stale += [f"{f.relative_to(ROOT)}: {s}" for s in STALE if s in text]
    assert not stale, stale


def test_removed_audiomoth_files_stay_removed():
    assert not [p for p in REMOVED if (ROOT / p).exists()]


def test_only_the_audiomoth_part_header():
    inc = ROOT / "audiomoth/vendor/audiomoth/device/inc"
    parts = sorted(p.name for p in inc.glob("efm32wg*.h") if not p.name.startswith("efm32wg_"))
    assert parts == ["efm32wg380f256.h"]
