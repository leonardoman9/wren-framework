# Contract: CI workflow (`.github/workflows/ci.yml`)

**Triggers**: push to `main`; every pull request. Plain checkout: the repository has no submodules.

| Job | Runner | Steps | Pass criteria |
|---|---|---|---|
| `docker` | `ubuntu-latest` | free disk space (unused toolchains) and log the runner's CPU, then `docker buildx build --platform linux/amd64` (GHA layer cache), then inside the image: `python tests/test_training.py`, `python -m pytest tests/test_setup.py tests/test_examples.py tests/test_stage_keys.py`, `python tests/make_synthetic_dataset.py runs/synth_data` + `python -m wren run tests/synthetic/project.yaml`, `tests/test_birds_reproduction.sh` | All exit 0; bird package files identical; `birds.bin` and `birds_diag.bin` sha256 match `tests/expected/birds_firmware.sha256` |
| `native-linux` | `ubuntu-24.04` (x86_64) | apt install the prerequisites, `actions/setup-python` 3.11, `python -m wren setup --tflm`, `python -m wren setup --check`, the synthetic end-to-end run | All exit 0; every parity check and quality gate passes; byte-identity not checked |
| `native-macos` | `macos-14` (arm64) | `brew install make wget mtools qemu` (GNU make 4 for TFLM), Python 3.11, then as `native-linux`; training falls back to the CPU when the runner's GPU cannot allocate | As `native-linux` |

**Caches**: pip wheels, keyed on the lock file hash. The TFLM directory (source, GCC download and built
libraries), keyed on the TFLM commit, the patch hash and `build_tflm.sh`. Docker layers via `type=gha`.

**Time budget**: each job finishes in under 20 minutes with warm caches (SC-003). A cold run (cache miss) is
allowed to exceed it. Measured on 2026-10-22 (run 39846599144, warm): docker 19.5 min, native-linux 10.9 min,
native-macos 14.0 min. Cold (TFLM rebuilt): docker 33.6 min, native-linux 18.1 min, native-macos 17.0 min.

**Failure contract**: A failing stage exits non-zero. The job log shows the stage's plain-language `FAILED:`
line, and `runs/**/logs` is uploaded as an artifact on failure. The docker job also uploads the bird
conversion's intermediate files on failure, to locate a byte difference between hosts (research R13).

**Synthetic dataset**: generated at job time by `tests/make_synthetic_dataset.py`; no audio is committed. The
epochs are set low enough to fit the budget while still passing the evaluate→export gate.
