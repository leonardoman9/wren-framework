# Implementation Plan: Reproducible from Zero

**Branch**: `001-reproducible-from-zero` | **Date**: 2026-10-22 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-reproducible-from-zero/spec.md`

## Summary

Today the pipeline only runs on the developer's Mac. `.venv-torch` and `.venv-tf` are symlinks into a
session scratchpad, the bird example reads its calibration audio from absolute paths under
`~/Progetti/BioDCASE` and `~/Downloads`, the firmware comes from a separate repository (`audiomoth-fw` submodule) checked out at a
commit that exists only on a side branch, and there is no CI.

The approach has three parts:

- **Bootstrap.** `python -m wren setup` becomes a standard-library-only command. It checks prerequisites,
  builds both environments from fully locked, hash-checked requirements, fetches and builds TFLM (which
  downloads its pinned GCC 14.3.1).
- **Reference path.** A Dockerfile pinned to a base-image digest, a Debian snapshot and `linux/amd64` is the
  byte-identical reference path on any host. Users and CI build it themselves.
- **Self-contained bird example.** BioDCASE is removed from the project entirely. The bird example is
  recalibrated on a committed bundle of 3 s clips from freely licensed xeno-canto recordings (CC0 / CC-BY /
  CC-BY-SA, per-clip attribution): the 8 model species first, then other European species, because only
  about 1% of xeno-canto is freely licensed. The regenerated package, without the
  BioDCASE test window, becomes the new reference.
- **One self-contained repository.** The firmware is copied in under `audiomoth/` (from audiomoth-nn-deployment
  at cfa7ced, without history and without the BioDCASE test vectors). WrenNet's code and the bird checkpoint go
  under `wrennet/` (without history or generated logs). Both submodules are removed, so a plain `git clone` is
  enough. The byte-exact reproduction then runs
  anywhere.

GitHub Actions runs three jobs: Docker (byte-exact reproduction, synthetic end-to-end and unit tests), native
Linux x86_64 and native macOS ARM. The two native jobs check the parity checks and quality gates.

## Technical Context

**Language/Version**: Python 3.11 (environments). The setup bootstrap uses the standard library only. POSIX sh
for build scripts; C/C++ for the firmware (unchanged).

**Primary Dependencies**: PyTorch 2.14.1, torchaudio 2.11.0, onnx 1.23.2 and onnxruntime 1.31.0 (torch env);
TensorFlow 2.16.2, onnx2tf 1.28.8 and ai-edge-litert 2.3.0 (tf env); TFLM at e0021ca plus `tflm-no-int4.patch`;
Arm GNU Toolchain 14.3.Rel1 (downloaded by TFLM); `uv` (developer-only, to regenerate lock files).

**Storage**: Files only: `runs/<name>/…`, `envs/lock/*.txt`, `examples/birds/calibration/` (FLAC,
`SHA256SUMS`, `ATTRIBUTION.csv`, `LICENSE.md`), `tests/expected/`.

**Testing**: `tests/test_training.py` (unit), `tests/synthetic/project.yaml` (end-to-end smoke),
`tests/test_birds_reproduction.sh` (byte-exact, extended to firmware), the firmware's
`scripts/run_tests.sh` (host, simulator, QEMU), and a new `tests/test_setup.py` (bootstrap checks).

**Target Platform**: Docker `linux/amd64` on any host (byte-identical reference); native Linux x86_64 and
macOS arm64 (supported, CI-tested, parity only); macOS x86_64 and other platforms are Docker-only (see research R11).

**Project Type**: CLI pipeline (Python package `wren`) plus firmware build; model code (`wrennet/`) and firmware (`audiomoth/`) in-repo, no submodules.

**Performance Goals**: CI under 20 min per job (SC-003); clone to `.bin` in under 30 min hands-on (SC-001).

**Constraints**: No machine-specific paths in the repo. Every download is pinned and checksum-verified.
Firmware fits in 245,760 B. User-facing messages contain no ML or toolchain jargon (Principle I).

**Scale/Scope**: Around 10 files changed in `wren/`, 1 Dockerfile, 1 workflow, at least 400 committed
calibration clips from xeno-canto (about 40 MB), 4 lock files, 1 regenerated bird package, the firmware
imported under `audiomoth/` (about 31 MB, 363 files), WrenNet code and checkpoint under `wrennet/` (about 2 MB), and removal of `examples/biodcase`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Biologist-First | PASS | Setup errors name the missing thing and the fix in plain words ("Python 3.11 is needed: …", "Docker is the supported way on this Mac: …"). Toolchain and TFLM details go to `runs/…/logs` and the developer docs only. |
| II. One YAML, One Command | PASS | `python -m wren setup` is a one-time install step. The pipeline is still one `project.yaml` plus `python -m wren run`. `examples/birds/project.yaml` loses its absolute paths and uses only existing config keys. |
| III. Fixed Architecture | PASS | Untouched. The bird model keeps its float weights; only the int8 ranges change with the new calibration audio. |
| IV. End-to-End Verification | PASS | Every native and Docker run still executes every parity check and the firmware tests. The bird reproduction is extended from the package to the firmware `.bin` hash. |
| V. Reproducibility | PASS (constitution 2.0.0) | Clarification 2 limits byte-identity to the Docker path. Constitution v1.0.0 promised it on every platform, so it was amended to 2.0.0 on 2026-10-22: byte-identical on the Docker reference path; native platforms must pass all parity checks and gates; the bird reproduction runs in CI inside Docker. The training tolerance is documented (research R9). |
| VI. Honesty in Reporting | PASS (with addition) | The report records which path produced it: the Docker image ID, or "native install: outputs verified by parity checks, not byte-reproducible". |
| Pipeline & Quality Gates | PASS | Gates unchanged; CI exercises all of them. |

**Post-design re-check (after Phase 1)**: all principles pass against constitution 2.0.0. No violations were introduced
by the contracts or the data model.

## Project Structure

### Documentation (this feature)

```text
specs/001-reproducible-from-zero/
├── plan.md              # This file
├── research.md          # Phase 0: decisions R1–R12
├── data-model.md        # Phase 1: calibration bundle, locks, reference hashes, setup checks
├── quickstart.md        # Phase 1: validation scenarios
├── contracts/
│   ├── cli-setup.md         # python -m wren setup: flags, checks, exit codes, messages
│   ├── calibration-bundle.md # bundle layout, builder, project.yaml keys, BioDCASE exclusion
│   ├── docker-image.md      # build/run contract, pinned inputs
│   └── ci-workflow.md       # jobs, triggers, pass criteria, time budget
└── tasks.md             # Phase 2 (/speckit-tasks). The current file predates the clarifications: regenerate it.
```

### Source Code (repository root)

```text
wren/
├── __main__.py          # unchanged entry
├── cli.py               # lazy imports: `setup` must not import yaml/runner/stages
├── setup.py             # NEW: stdlib-only bootstrap (checks, venvs from locks, TFLM, --check)
├── stages/export.py     # audio_files: also *.flac
├── torchside/make_specs.py  # glob: also *.flac
└── stages/report.py     # records reproduction path (Docker image ID / native)

envs/
├── requirements-torch.in, requirements-tf.in   # top-level intent (renamed from the current .txt)
└── lock/{torch,tf}-{linux-x86_64,macos-arm64}.txt  # full transitive pins with hashes

docker/Dockerfile        # digest-pinned base, --platform=linux/amd64, snapshot.debian.org, TFLM built in
.dockerignore            # NEW: runs/, .venv-*, third_party/**/tflite-micro
.github/workflows/ci.yml # NEW: docker, native-linux, native-macos jobs

examples/birds/
├── project.yaml         # calibration.audio / test.audio: [calibration/audio] (no absolute paths)
└── calibration/         # NEW: audio/*.flac (xeno-canto CC0/BY/BY-SA, 32 kHz mono, 3 s), SHA256SUMS, ATTRIBUTION.csv, LICENSE.md, README.md
examples/biodcase/       # REMOVED (FR-014)

tests/
├── test_birds_reproduction.sh   # + firmware stage, compares .bin sha256 with tests/expected/
├── expected/birds_firmware.sha256  # NEW: from the Docker reference run
├── test_setup.py        # NEW: bootstrap checks (python version, disk, symlink replacement)
├── test_examples.py     # NEW: no absolute paths, no BioDCASE in any example/test YAML
├── other_classes/project.yaml  # calibration audio → examples/birds/calibration/audio

audiomoth/                # NEW (FR-016): firmware copied from audiomoth-nn-deployment@cfa7ced, no history;
                         # firmware/bird/model regenerated (pipe_vec.cc, specs_one.npz without BioDCASE),
                         # e2e_make_ref.py glob *.flac, docs/VERIFICATION.md §9 marked as previous package
wrennet/                 # NEW (FR-017): WrenNet code + best_distillation_model.pt, no history, no logs/benchmarks
third_party/             # REMOVED: both submodules (audiomoth-fw, WrenNet) and .gitmodules
wren/config.py, wren/runner.py, wren/torchside/model.py, wren/stages/{data,package}.py
                         # paths → audiomoth/ and wrennet/; stage keys hash these folders instead of submodule commits
.specify/memory/constitution.md  # Principle V amendment (2.0.0)
docs/PIPELINE.md         # NEW (FR-019): the technical stage-by-stage description moved out of README.md
docs/PLAN_STEP3.md       # REMOVED (FR-020): finished planning document
tools/make_calibration_bundle.py  # MOVED from tests/ (FR-020): maintainer tool
tests/test_training.py   # RENAMED from tests/test_step3.py (FR-020)
audiomoth/.github/, audiomoth/benchmarks/  # REMOVED (FR-020): the old repo's CI and Raspberry Pi benchmarks
README.md                # FR-019: for biologists — setup, platforms, Docker, reproducibility, project.yaml, results,
                         # on the device (flash, CONFIG.TXT, log), licences and credits; no BioDCASE
```

**Structure Decision**: Single project, matching the existing layout (`wren/`, `tests/`, `docker/`, `envs/`,
`examples/`). No new top-level packages.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Principle V narrowed to the Docker path | TF/onnx2tf float conversion and the GCC host build differ by CPU architecture and OS. Byte-identity across native macOS ARM and Linux x86_64 is not achievable with the pinned tools (clarification 2). | Byte-identity everywhere would require running the TFLite conversion and compiler inside a single emulated architecture, which is what the Docker path already is. |
| Committed calibration bundle (about 40 MB, xeno-canto CC0/BY/BY-SA) | The int8 bytes depend on the exact calibration clips (clarification 1). BioDCASE may not be used anywhere. | Synthetic audio gives int8 ranges that do not represent birds. Downloading at run time would make CI depend on xeno-canto and an API key. |
