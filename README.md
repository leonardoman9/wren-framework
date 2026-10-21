# wren-framework

From a trained **WrenNet** model to **AudioMoth** firmware, automatically. A YAML file says which checkpoint, which
classes and how the device should listen; one command converts the model (PyTorch → ONNX → int8 TFLite), generates the
C code the firmware needs, builds the firmware, runs its tests and writes a report.

```sh
python -m wren run examples/birds/project.yaml
```

```
runs/birds/
  export/     the int8 backbone and everything used to make it (ONNX, SavedModel, calibration spectrograms)
  package/    the model package the firmware builds from (int8 backbone, generated C headers, class list, test vector)
  firmware/   birds.bin (flash it with the AudioMoth Flash App) and birds_diag.bin (diagnostic recorder)
  report.md   what was made and how well it matches the checkpoint
  logs/       the output of every step
```

Status: **step 2 of the plan** (conversion of a trained model). Training from audio folders (`prepare`, `train`,
`evaluate`) comes next; then the project file will name audio folders instead of a checkpoint.

## What it does

| stage | steps | checks |
|---|---|---|
| `export` | split the checkpoint into front-end + core; backbone to ONNX in 4-D (no RESHAPE, `Hardswish` rewritten for TFLM); onnxsim; onnx2tf; int8 TFLite calibrated on your audio | front-end + core = whole model; ONNX = PyTorch; only operators the firmware registers; int8 vs float error |
| `package` | learned filterbank → C tables; GRU/attention/classifier → int8 C weights; classes and device settings → `model_config.h`; one test window with its expected logits | the device pipeline (C front-end + int8 backbone + C tail, built for the computer) against PyTorch on test windows |
| `firmware` | build the normal and the diagnostic firmware | flash budget; the firmware repository's tests on this package: schedule logic, FatFS, simulator with 44 fault scenarios, pipeline on an emulated Cortex-M4F |
| `report` | `report.md` | — |

Each stage is skipped when its inputs did not change (`--force` reruns it); `--stages export,package` runs a subset.

**Reproducibility.** `tests/test_birds_reproduction.sh` runs `examples/birds` and checks that the result is, byte for
byte, the bird package shipped in the firmware repository, which was made and verified by hand: the int8 model
(sha256 `6f77fa86…`), every generated header, the test vector. The converter is sensitive to library versions, so the
environments are pinned (`envs/`).

## Setup

Needs Python 3.11, a C compiler, git; for the firmware stage also `mtools` and `qemu-system-arm` (tests), and TFLM.

```sh
git clone --recursive <this repository>      # or: git submodule update --init
python3.11 -m venv .venv-torch && .venv-torch/bin/pip install -r envs/requirements-torch.txt
.venv-torch/bin/python -m wren setup --tflm  # the TensorFlow environment, and TFLM for the firmware (~10 minutes)
.venv-torch/bin/python -m wren run examples/birds/project.yaml
```

(`wren` itself only needs PyYAML; running it with the torch environment's Python is the simplest.) The conversion runs
in two environments because PyTorch and TensorFlow need different `numpy` versions: `.venv-torch` (export, reference
data, C generators) and `.venv-tf` (onnx2tf, int8 conversion, TFLite checks). `docker/Dockerfile` builds both, with
TFLM and the firmware toolchain.

## project.yaml

```yaml
name: birds                                  # output folder runs/<name>
checkpoint: path/to/model.pt                 # a WrenNet checkpoint (any number of classes, architecture fixed)
classes: [Species A, Species B, background]  # names in the model's OUTPUT order; the count must match the checkpoint
negative: last                               # the "nothing of interest" class: index, last or none
calibration:
  audio: [folder/, file.wav]                 # audio like the deployment's (first 3 s of each file, shuffled, seed 0)
  count: 300                                 # spectrograms for the int8 calibration
  eval: 100                                  # spectrograms to measure the int8 error
test:
  audio: [folder/]                           # windows for the device-vs-PyTorch check (default: the calibration audio)
  windows: 120
device:
  window_start: "07:00"                      # daily recording window, device clock
  window_minutes: 5
  hop_seconds: 3                             # step between classified 3 s windows, 1-3 s
  confidence: 50                             # % above which a detection is "confident" (Y in the log, green LED)
firmware:                                    # optional
  tests: true
```

Relative paths are relative to the YAML file. The checks that do not need the model (paths, class count, names,
window) run first, so a mistake costs seconds.

## Layout

```
wren/              the pipeline (cli.py, config.py, runner.py, stages/)
wren/torchside/    model loading (any class count), reference data; export_backbone.py and make_specs.py are
                   WrenNet's conversion scripts, unchanged, run with the general loader
wren/tfside/       int8 conversion
envs/              pinned requirements of the two environments
examples/birds/    the WrenNet bird model (8 species + non-bird)
tests/             reproduction test
third_party/       WrenNet (architecture, checkpoint) and audiomoth-fw (firmware), as git submodules
```
