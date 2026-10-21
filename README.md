# wren-framework

From audio folders, or from a trained **WrenNet** model, to **AudioMoth** firmware, automatically. A YAML file names
one folder per class (any animals, any number of classes), or a checkpoint; one command trains the model with the
WrenNet architecture, evaluates it the way the device will use it, converts it (PyTorch → ONNX → int8 TFLite),
generates the C code the firmware needs, builds the firmware, runs its tests and writes a report.

```sh
python -m wren run my_project.yaml          # audio folders -> trained model -> firmware
python -m wren run examples/birds/project.yaml   # an existing checkpoint -> firmware
```

```
runs/<name>/
  prepare/    the audio, decoded to 32 kHz and split by file; which 3 s windows contain the animal (training projects)
  train/      best.pt, history.csv, train.json (training projects)
  evaluate/   evaluate.json, confusion.csv: the test split, per window and per file with the device's rule
  export/     the int8 backbone and everything used to make it (ONNX, SavedModel, calibration spectrograms)
  package/    the model package the firmware builds from (int8 backbone, generated C headers, class list, test vector)
  firmware/   <name>.bin (flash it with the AudioMoth Flash App) and <name>_diag.bin (diagnostic recorder)
  report.md   what was made and how well it works
  logs/       the output of every step
```

## What it does

| stage | steps | checks |
|---|---|---|
| `prepare` | find the audio (wav, flac, ogg, mp3, aiff, any case; files up to 30 min), split by file (or subfolder) with a fixed seed, decode once to 32 kHz with the device's DC filter; for each positive file, the 3 s windows with an event in the class's frequency band; background windows on the device's grid | errors for: fewer than 5 files in a class, empty splits, duplicates across splits, names the firmware cannot store, >10 % unreadable; warnings for: different bandwidths or sample rates between classes, files without a clear event, padding |
| `train` | WrenNet architecture, the class count from the folders; from WrenNet's weights (`init: wrennet`, learned filterbank kept) or from scratch; background share per batch, augmentation (gain, background mixing, speed, low-pass, SpecAugment); early stopping on validation macro-F1 | the learned filterbank stays valid |
| `evaluate` | test split: per call window, and per file with the device's rule (every window on its grid, threshold); threshold suggested on validation; false reports per hour of background | macro-F1 must beat chance |
| `export` | split the checkpoint into front-end + core; backbone to ONNX in 4-D (no RESHAPE, `Hardswish` rewritten for TFLM); onnxsim; onnx2tf; int8 TFLite calibrated on your audio | front-end + core = whole model; ONNX = PyTorch; only operators the firmware registers; int8 vs float error |
| `package` | learned filterbank → C tables; GRU/attention/classifier → int8 C weights; classes and device settings → `model_config.h`; one test window with its expected logits | the device pipeline (C front-end + int8 backbone + C tail, built for the computer) against PyTorch on test windows; for trained models, int8 macro-F1 at most 0.05 below float |
| `firmware` | build the normal and the diagnostic firmware | flash budget; the firmware repository's tests on this package: schedule logic, FatFS, simulator with 44 fault scenarios, pipeline on an emulated Cortex-M4F |
| `report` | `report.md` | — |

Each stage is skipped when its inputs did not change (`--force` reruns it); `--stages export,package` runs a subset.

**Reproducibility.** `tests/test_birds_reproduction.sh` runs `examples/birds` and checks that the result is, byte for
byte, the bird package shipped in the firmware repository, which was made and verified by hand: the int8 model
(sha256 `6f77fa86…`), every generated header, the test vector. The converter is sensitive to library versions, so the
environments are pinned (`envs/`).

## Measured so far

| run | data | test macro-F1, call windows (float → int8 on the device pipeline) |
|---|---|---|
| `tests/synthetic` | 3 synthetic tonal "species" + background, from scratch | 1.000 → 1.000; firmware built and tested |
| `examples/biodcase`, `init: wrennet`, seed 0 | BioDCASE 2026 TinyML development set: 10 species + background, 3 s clips at 24 kHz | 0.557 → 0.552 (chance 0.091); firmware built, all its tests pass |
| same, seed 1 | | 0.557 |
| same, `init: scratch` | | 0.488 |

Limits of these numbers: BioDCASE is 3 s clips recorded at 24 kHz (nothing above 12 kHz, while the AudioMoth records up
to 16 kHz), and its background split is 2.5 minutes, so the false reports per hour it gives mean nothing. No
long field recordings from an AudioMoth have been used yet: the window selection on long files is tested on synthetic
data only (`tests/test_step3.py`), and false reports per hour need hours of background from the deployment sites.

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

A training project names audio folders instead of a checkpoint (the classes come from the folder names):

```yaml
name: my-frogs
data:
  root: recordings/              # recordings/<class>/*.wav, one folder per class; split into train/validation/test by file
  # or: train: ..., validation: ..., test: ...   (folders with the same class subfolders; missing ones are carved from train)
  negative: background           # the folder with no animal of interest (recommended: background of your sites)
  names: {hyla_arborea: Tree frog}   # optional display names (ASCII, max 40 characters, no commas or quotes)
  group: file                    # or subfolder: keep files of one recorder/site/day (a subfolder) in one split
  split: {validation: 0.15, test: 0.15, seed: 0}
  windows: {hop_seconds: 1.5, max_per_file: 6, negative_max_per_file: 40}
training:                        # all optional; defaults shown
  init: wrennet                  # or scratch (learns the filterbank too; needs more data)
  epochs: 40
  patience: 8
  batch_size: 32
  lr: 5.0e-4
  negative_share: 0.4            # share of background windows per batch
  augment: true
  loss: ce                       # or focal
  seed: 0
device: {window_start: "05:30", window_minutes: 10, hop_seconds: 3, confidence: 50}
```

Calibration and test windows for the int8 conversion come from the training and validation data. Long recordings are
fine: the label is per file, so `prepare` keeps only the windows where something stands out in the class's frequency
band (wind or rain that every class has does not count). Files where nothing stands out are listed in
`prepare.json`.

Relative paths are relative to the YAML file. The checks that do not need the model (paths, class count, names,
window) run first, so a mistake costs seconds.

## Layout

```
wren/              the pipeline (cli.py, config.py, runner.py, stages/)
wren/torchside/    prepare, train, evaluate; model loading (any class count), reference data; export_backbone.py
                   and make_specs.py are WrenNet's conversion scripts, unchanged, run with the general loader
wren/tfside/       int8 conversion
envs/              pinned requirements of the two environments
examples/birds/    the WrenNet bird model (8 species + non-bird)
examples/biodcase/ training on the BioDCASE 2026 TinyML development set (10 species + background)
tests/             reproduction test, training checks (test_step3.py), synthetic end-to-end (tests/synthetic)
third_party/       WrenNet (architecture, checkpoint) and audiomoth-fw (firmware), as git submodules
```
