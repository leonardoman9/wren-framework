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
| `train` | WrenNet architecture with the class count taken from the folders; starts from WrenNet's weights (`init: wrennet`, learned filterbank kept) or from scratch; background share per batch; augmentation (gain, background mixing, speed, low-pass, SpecAugment); early stopping on validation macro-F1. With `birdnet: true`: BirdNET v2.4 (TFLite, `ai_edge_litert`) scores the training audio on prepare's grid, resampled to 48 kHz (cached in `train/birdnet/`, keyed by file contents); loss = focal distillation, (1 − α)·focal + α·T²·KL, with the target corrected from the original WrenNet project (`wren/torchside/birdnet_target.py`, specs/003-birdnet-distillation research R6); the KL term is skipped on species windows where BirdNET gives every project species < 0.1; validation and model selection on folder labels only | the learned filterbank stays valid; a sound that is not a BirdNET species is refused in the project check |
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
| BirdNET (optional) | v2.4 TFLite, Zenodo 15050749, zip and both files by sha256 (`wren/birdnet.py`); `third_party/birdnet/`, not committed |

Byte-identity is guaranteed only on the Docker path, which is `linux/amd64` on any host: the float conversion and
the compiler builds differ by CPU architecture and OS (constitution V). Native installs are held to the parity checks
and gates. `runs/<name>/provenance.json` records which path, platform, locks and source hashes made a run.

## Why the bytes match on every x86 computer

The first CI runs found that Docker on an AMD server and Docker emulated on Apple silicon gave a different int8
model and firmware, although every check passed on both. Numerical libraries pick their code from the CPU, and
CPUs round the last bit differently. Each source was removed where it was found:

| Source | Fix |
|---|---|
| PyTorch, MKL, oneDNN and TensorFlow kernels chosen by CPU features and thread count | `export` and `package` run with fixed kernels and one thread (`wren/runner.py`, `PINNED`) |
| MKL's last bit differs between Intel and AMD, and its reproducibility settings are ignored on AMD | batch norm is folded into the convolutions by `export_backbone.py`, in float64 with NumPy and rounded once to float32; calibration spectrograms and test-window logits are computed in float64 and rounded to float32 |
| TensorFlow's approximate `1/sqrt` instruction differs between Intel and AMD | the batch norms after residual sums are exported as Mul and Add with constants computed in float64 |
| TFLite's calibration runs the float model with kernels chosen from the CPU (AVX-512 on recent Intel, AVX2 on AMD and emulated) | the converter only lists the tensors that need a range; `wren/tfside/quantize.py` measures each range with TFLite's reference kernels (the tensor's actual min and max) and then runs the converter's quantizer |
| TFLM's Makefile ordered the library's objects as the filesystem listed them, which changed the firmware's layout | `audiomoth/firmware/patches/tflm-sorted-objects.patch`; the archives and the object folder are rebuilt from empty |

`tests/test_birds_reproduction.sh` in the CI's Docker job checks the result on every change.

## Tests

| Test | What it covers |
|---|---|
| `tests/test_training.py` | prepare, window selection, splits, error paths, model loading |
| `tests/test_setup.py` | `python -m wren setup` checks and messages, symlinked environments, idempotency |
| `tests/test_examples.py` | no absolute or BioDCASE paths in any project; the calibration bundle's attribution and licences |
| `tests/test_stage_keys.py` | a stage reruns when its audio (by content), its conversion code or the model code (`wrennet/**/*.py`, `wren/torchside/model.py`) changes; READMEs and caches do not count |
| `tests/test_birdnet.py` | learning from BirdNET: species matching, refusals, stored results without BirdNET, the corrected target and the loss |
| `tests/synthetic/project_birdnet.yaml` | the refusal: non-bird sounds with `birdnet: true` stop in the project check |
| `tests/birds_birdnet/project.yaml`, `project_off.yaml` | a bird project from the calibration bundle (`tests/make_bird_dataset.py`) with and without BirdNET, to the firmware |
| `tests/test_measure.py`, `tests/test_bird_accuracy.sh` | the bird accuracy statistics (bootstrap per recording, false detections); the measurement on `examples/birds/test/` reproduces `accuracy.json` (byte for byte in Docker) |
| `tests/test_layout.py` | `wrennet/` holds only the model package and the checkpoint; no old WrenNet paths; removed firmware files stay removed |
| `tests/synthetic/project.yaml` | the whole chain, from synthetic audio to tested firmware |
| `tests/minimal/project.yaml` | a project with only `data:` (every default) |
| `tests/other_classes/project.yaml` | a 4-class checkpoint: the pipeline is not tied to the bird model |
| `tests/test_birds_reproduction.sh` | `examples/birds` gives the shipped package byte for byte, and in Docker the firmware hashes in `tests/expected/` |

CI (`.github/workflows/ci.yml`) runs them in the Docker image, natively on Linux x86_64, and natively on macOS arm64.

## Learning from BirdNET: measured so far

`tests/birds_birdnet/` (8 species, at most 10 recordings each, 3 s calibration clips; background = 62 other species;
74 test windows), defaults α 0.5, T 4, γ 2, 40 epochs, 2026-10-23:

| init | | where | validation macro-F1 | test macro-F1 |
|---|---|---|---|---|
| wrennet | with BirdNET | macOS arm64 (mps) | 0.725 | 0.587 |
| wrennet | with BirdNET | Docker (CPU, CI) | 0.727 | 0.787 |
| wrennet | without | macOS arm64 (mps) | 0.749 | 0.801 |
| scratch | with BirdNET | macOS arm64 (mps) | 0.384 | 0.409 |
| scratch | without | macOS arm64 (mps) | 0.487 | 0.413 |

No conclusion can be drawn from these. The same project with BirdNET gave test macro-F1 0.587 on one computer and 0.787
on another (training is not byte-reproducible across computers): the run-to-run spread is as large as the difference
with and without BirdNET. The test set is too small (74 windows, mostly background, 4–8 per species), the clips come
from the calibration bundle, α and T were not tuned, and with `init: wrennet` the starting model was already trained
with BirdNET as teacher on these species. BirdNET named the folder's species on 88 % of the species training windows.
A meaningful comparison needs a larger labelled test set and several seeds; `examples/birds/test/` (feature 006) is
that set for the species and the false detections.

## Bird model accuracy

`python tools/measure_bird_accuracy.py` measures the shipped model on `examples/birds/test/` (freely licensed: xeno-canto
species recordings outside the calibration clips, BirdNET-screened CC0 Freesound field sounds) through prepare's
call-finding, the float reference and the device pipeline; figures, intervals and limits in
`audiomoth/docs/VERIFICATION.md` §9. The bundle is built once by `tools/make_test_bundle.py` (xeno-canto and Freesound
keys).

## Layout

```
wren/              the pipeline: cli.py, setup.py (stdlib only), config.py, runner.py, stages/
wren/torchside/    prepare, train, evaluate, model loading, reference data; export_backbone.py and make_specs.py
                   come from WrenNet's conversion scripts (make_specs.py also reads .flac)
wren/tfside/       int8 conversion and device-pipeline evaluation
wrennet/           WrenNet: the architecture and learned front-end (package `wrennet`), checkpoints/birds.pt
audiomoth/         the AudioMoth firmware, its tests and the TFLM build scripts (MIT, Open Acoustic Devices)
envs/              the environments' intent (.in) and locks
docker/            the reference Dockerfile
examples/birds/    the bird model project and its calibration bundle
tests/             tests and test projects (above)
tools/             maintainer tools (make_calibration_bundle.py)
specs/             feature specifications, plans and tasks (Spec Kit)
```
