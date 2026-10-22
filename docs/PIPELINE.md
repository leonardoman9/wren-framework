# How the pipeline works (technical)

The README describes wren-framework for its users. This page is for developers: what every stage does, what it
checks, and what makes a run reproducible.

## Stages

`python -m wren run PROJECT.yaml` runs these in order. Each stage writes `runs/<name>/<stage>/` with a `stage.json`
key: a hash of the stage's inputs, by content for files. A stage whose key has not changed is skipped. Use
`--force` to rerun stages, and `--stages export,package` to run a subset. Every step writes its command and output to
`runs/<name>/logs/<step>.log`; the console only shows plain-language step labels (constitution I).

| stage | steps | checks |
|---|---|---|
| `prepare` | finds the audio (wav, flac, ogg, mp3, aiff, any case; files up to 30 min); splits it by file (or subfolder) with a fixed seed; decodes once to 32 kHz with the device's DC filter; for each positive file, picks the 3 s windows with an event in the class's frequency band; picks background windows on the device's grid | errors: fewer than 5 files in a class, empty splits, duplicates across splits, names the firmware cannot store, >10 % unreadable files. Warnings: different bandwidths or sample rates between classes, files without a clear event, padding |
| `train` | WrenNet architecture with the class count taken from the folders; starts from WrenNet's weights (`init: wrennet`, learned filterbank kept) or from scratch; background share per batch; augmentation (gain, background mixing, speed, low-pass, SpecAugment); early stopping on validation macro-F1 | the learned filterbank stays valid |
| `evaluate` | test split, per call window and per file with the device's rule (every window on its grid, threshold); threshold suggested on validation; false reports per hour of background | **gate:** test macro-F1 must exceed 1/N + 0.1 |
| `export` | splits the checkpoint into front-end + core; exports the backbone to ONNX in 4-D (no RESHAPE, `Hardswish` rewritten for TFLM); onnxsim; onnx2tf; int8 TFLite calibrated on the project's audio | front-end + core = whole model; ONNX = PyTorch; only operators the firmware registers; int8 vs float error |
| `package` | learned filterbank → C tables; GRU/attention/classifier → int8 C weights; classes and device settings → `model_config.h`; one test window with its expected logits | the device pipeline (C front-end + int8 backbone + C tail, built for the host) against PyTorch on test windows; **gate (trained models):** int8 macro-F1 at most 0.05 below float |
| `firmware` | builds the normal and the diagnostic firmware in `runs/<name>/firmware_build/` | flash budget (245,760 bytes); the firmware's tests on this package: schedule logic, FatFS, simulator with sanitizers and 44 fault scenarios, pipeline on an emulated Cortex-M4F (QEMU). `<name>.bin` reaches `firmware/` only after all of them pass. With `firmware: tests: false` it is named `<name>_UNTESTED.bin` (constitution IV) |
| `report` | `report.md` (plain summary, then technical details), `provenance.json` | — |

## Two Python environments

PyTorch and TensorFlow need different `numpy` versions:
- `.venv-torch`: prepare, train, evaluate, ONNX export, reference data, the C generators;
- `.venv-tf`: onnx2tf, int8 conversion, the TFLite and device-pipeline checks.

`wren/runner.py` runs each step in the right one. `python -m wren run` re-runs itself inside `.venv-torch` when the
Python that was typed lacks the pipeline's packages, so users never call a venv directly.

## What is pinned

| Input | Pinned by |
|---|---|
| Python packages | `envs/lock/{torch,tf}-{linux-x86_64,macos-arm64}.txt`, every package with hashes (`envs/README.md`) |
| TFLM | commit `e0021ca` + the local patches `audiomoth/firmware/patches/*.patch` (int4 kernels out; object lists sorted, so the firmware layout is the same everywhere) |
| ARM compiler | Arm GNU Toolchain 14.3.Rel1 (GCC 14.3.1), downloaded and MD5-checked by TFLM's own script |
| Docker base image | `python:3.11.9-slim-bookworm` by digest, `linux/amd64` |
| Debian packages | `snapshot.debian.org` at a fixed date (`docker/Dockerfile`, `DEBIAN_SNAPSHOT`) |
| Model code and checkpoint | `wrennet/` (in this repository) |
| Firmware | `audiomoth/` (in this repository) |
| Bird calibration audio | `examples/birds/calibration/` with `SHA256SUMS` |

Byte-identity is guaranteed only on the Docker path, which is `linux/amd64` on any host: the float conversion and
the compiler builds differ by CPU architecture and OS (constitution V). Native installs are held to the parity checks
and gates. `runs/<name>/provenance.json` records which path, platform, locks and source hashes made a run.

## Tests

| Test | What it covers |
|---|---|
| `tests/test_training.py` | prepare, window selection, splits, error paths, model loading |
| `tests/test_setup.py` | `python -m wren setup` checks and messages, symlinked environments, idempotency |
| `tests/test_examples.py` | no absolute or BioDCASE paths in any project; the calibration bundle's attribution and licences |
| `tests/synthetic/project.yaml` | the whole chain, from synthetic audio to tested firmware |
| `tests/minimal/project.yaml` | a project with only `data:` (every default) |
| `tests/other_classes/project.yaml` | a 4-class checkpoint: the pipeline is not tied to the bird model |
| `tests/test_birds_reproduction.sh` | `examples/birds` gives the shipped package byte for byte, and in Docker the firmware hashes in `tests/expected/` |

CI (`.github/workflows/ci.yml`) runs them in the Docker image, natively on Linux x86_64, and natively on macOS arm64.

## Layout

```
wren/              the pipeline: cli.py, setup.py (stdlib only), config.py, runner.py, stages/
wren/torchside/    prepare, train, evaluate, model loading, reference data; export_backbone.py and make_specs.py
                   come from WrenNet's conversion scripts (make_specs.py also reads .flac)
wren/tfside/       int8 conversion and device-pipeline evaluation
wrennet/           WrenNet: architecture, original training code, the bird checkpoint
audiomoth/         the AudioMoth firmware, its tests and the TFLM build scripts (MIT, Open Acoustic Devices)
envs/              the environments' intent (.in) and locks
docker/            the reference Dockerfile
examples/birds/    the bird model project and its calibration bundle
tests/             tests and test projects (above)
tools/             maintainer tools (make_calibration_bundle.py)
specs/             feature specifications, plans and tasks (Spec Kit)
```
