# wren-framework

From your audio recordings to firmware for an **AudioMoth** that recognises your species on the device.

You put your recordings in one folder per species (or other sound), write a short `project.yaml`, and run one
command. wren-framework then does everything else:
- trains a **WrenNet** model (a small neural network for animal sounds);
- tests it on recordings it has not seen;
- shrinks it to fit the AudioMoth;
- builds the firmware and tests it on a simulated device;
- writes a report you can attach to a paper or a funding application.

It also converts an already trained WrenNet model, such as the bird model that ships with it.

```sh
python3.11 -m wren setup --tflm                      # once: installs what the pipeline needs
python3.11 -m wren run my_project.yaml               # your recordings -> trained model -> firmware
python3.11 -m wren run examples/birds/project.yaml   # the shipped bird model -> firmware
```

## What you get

```
runs/<name>/
  report.md   what was made, whether it fits on the device, and how well it works (start here)
  firmware/   <name>.bin to flash on the AudioMoth, and <name>_diag.bin (a diagnostic version)
  evaluate/   accuracy on your test recordings, per sound and per file (training projects)
  train/      the trained model (training projects)
  logs/       the details of every step, for when something goes wrong
```

## What happens, step by step

1. **Prepare:** reads your recordings, finds the 3-second moments where the animal calls, and splits the files
   into training, validation and test sets. Files from the same recorder can stay together, and no file is ever
   in two sets.
2. **Train:** teaches the model your sounds, starting from what WrenNet already knows about birdsong. It stops
   when the validation recordings stop improving.
3. **Evaluate:** measures accuracy on the test recordings, the way the device will see them, and suggests a
   confidence threshold. If the model is barely better than guessing, it stops and says so.
4. **Convert:** shrinks the model to fit the AudioMoth's small memory, then checks that the small version gives
   the same answers as the original.
5. **Firmware:** builds the AudioMoth firmware and runs its tests on a simulated device, including two days of
   recording and 44 kinds of fault (no SD card, power cut, full card…). The firmware file is delivered only if
   every test passes.
6. **Report:** summarises all of the above in plain words, then gives the technical details.

Every step is checked, and a failed check stops the run with a message saying what went wrong. A step that has
already run is skipped when nothing it depends on has changed. The technical description of each step is in
[docs/PIPELINE.md](docs/PIPELINE.md).

## Setup

What the computer needs:

- **Python 3.11** (python.org, Homebrew or your package manager).
- **A few programs.**
  - On Linux: `sudo apt install git make gcc clang wget mtools qemu-system-arm`.
  - On macOS: `xcode-select --install`, then `brew install make wget mtools qemu`.
- **About 6 GB of free disk** for the tools (they use about 4 GB), plus room for your runs, and **8 GB of RAM**.

```sh
git clone <this repository> && cd wren-framework
python3.11 -m wren setup --tflm     # about 3 minutes on a recent Mac; run it again any time, it only fixes what is missing
python3.11 -m wren setup --check    # says what is ready and what is missing
```

`setup` checks the computer first and says, in plain words, what is missing and how to install it. It then
installs everything at fixed, verified versions into the project folder; nothing is installed system-wide. If
something fails, the details are in `runs/_setup/logs/setup.log`.

You can type `python3.11 -m wren run …` (or `python -m wren run …`) with the same Python you used for `setup`:
`run` finds the installed tools by itself.

### Supported computers

| Computer | How | Checked automatically |
|---|---|---|
| Linux, Intel/AMD 64-bit | `setup` as above | yes |
| Mac with Apple silicon (M1 or later), **macOS 14 or later** | `setup` as above | yes |
| Any computer with Docker (Windows, Intel Macs, other Linux) | Docker, below | yes |

On an Intel Mac, `setup` stops right away and points to Docker, because the analysis libraries no longer support
it directly.

### Using Docker

```sh
docker build --platform linux/amd64 -f docker/Dockerfile -t wren .
docker run --rm --platform linux/amd64 -v "$PWD/runs:/wren/runs" -v /path/to/recordings:/audio wren run /wren/examples/birds/project.yaml
```

Docker builds the same environment on every computer. On a Linux PC the build takes a few minutes. On a Mac with
Apple silicon it is emulated and takes about 20 minutes, but only the first time.

### What is reproducible

- **With Docker:** the same project, recordings and version of this repository give **byte-identical** model and
  firmware files on any computer. `tests/test_birds_reproduction.sh` checks this for the bird model on every
  change.
- **With `setup` on Linux or a Mac:** the results pass every check above, but their bytes may differ slightly from
  Docker's, because computers with different processors round numbers slightly differently.
- **Every report says which of the two made it.**
- **Training is not compared byte for byte.** On the same computer, with the same settings, it repeats. On another
  computer or a graphics card the trained weights differ. A trained model is accepted when it passes the quality
  checks, not when its bytes match:
  - test macro-F1 above 1/N + 0.1, for N sounds;
  - the device version within 0.05 macro-F1 of the original.

## Your project: project.yaml

A training project names your recordings. The sounds are the folder names, so this is enough:

```yaml
data:
  root: recordings/              # recordings/<sound>/*.wav, one folder per sound
  negative: background           # the folder with no animal of interest (record it at your sites)
```

Everything else has a default. These are all the options, with their defaults:

```yaml
name: my-frogs                   # output folder runs/<name> (default: the YAML's folder name)
data:
  root: recordings/              # or: train: ..., validation: ..., test: ...  (folders with the same sound subfolders;
                                 #     missing ones are taken from train)
  negative: background
  names: {hyla_arborea: Tree frog}   # optional display names (ASCII, max 40 characters, no commas or quotes)
  group: file                    # or subfolder: keep the files of one recorder/site/day (a subfolder) in one set
  split: {validation: 0.15, test: 0.15, seed: 0}
  windows: {hop_seconds: 1.5, max_per_file: 6, negative_max_per_file: 40}
training:
  init: wrennet                  # or scratch (learns everything from your data; needs more of it)
  epochs: 40
  patience: 8                    # stop after this many epochs without improvement
  batch_size: 32
  lr: 5.0e-4
  negative_share: 0.4            # share of background in each training batch
  augment: true                  # vary loudness, background, speed and frequency while training
  loss: ce                       # or focal
  focal_gamma: 2.0               # with loss: focal
  seed: 0
  device: auto                   # cuda, mps or cpu; auto picks the fastest available
  warmup_epochs: 2
device:
  window_start: "05:30"          # daily recording window, device clock
  window_minutes: 10
  hop_seconds: 3                 # step between classified 3 s windows, 1-3 s
  confidence: 50                 # % above which a detection is "confident" (Y in the log, green LED)
```

Long recordings are fine. Each file's label is its folder, so `prepare` keeps only the moments where something
stands out in that sound's frequency range. Wind or rain that every folder has does not count. Files where nothing
stands out are listed in `runs/<name>/prepare/prepare.json`.

To convert an already trained WrenNet model instead (as `examples/birds/project.yaml` does):

```yaml
name: birds
checkpoint: path/to/model.pt                 # a WrenNet model (any number of sounds)
classes: [Species A, Species B, background]  # names in the model's output order
negative: last                               # the "nothing of interest" class: index, last or none
calibration:
  audio: [folder/]                           # recordings like your sites' (first 3 s of each file), used to shrink
  count: 300                                 #   the model and to measure what shrinking costs
  eval: 100
test:
  audio: [folder/]                           # recordings for the device-vs-original check (default: as above)
  windows: 120
device: {window_start: "07:00", window_minutes: 5, hop_seconds: 3, confidence: 50}
```

Advanced options:
- `reference_audio` (the clip for the reference results);
- `firmware: {tests: true, repo: ..., tflm: ...}`. `tests: false` is for developers only: the firmware files are
  then named `*_UNTESTED.bin` and must not be deployed;
- `envs: {torch: ..., tf: ...}`;
- `runs: ...` (where the output goes).

Paths are relative to the YAML file. The checks that do not need the model (paths, number of sounds, names,
recording window) run first, so a mistake costs seconds.

## On the device

1. Flash `runs/<name>/firmware/<name>.bin` with the **AudioMoth Flash App**, choosing it as a local firmware file.
2. Format a microSD card as FAT32. In its root, create `CONFIG.TXT` with the current time on one line, for example
   `2026-10-22 14:30:00`. The board sets its clock from it at boot.
3. Insert the card and power on. Every day, during the recording window, the board records to `YYYYMMDD/HHMMSS.WAV`.
   It then classifies the recording and writes one line per 3 s window to `LOG.TXT`:
   ```
   2026-10-21 07:00:09,Bubo bubo,69,Y,...
   time, most likely sound, confidence %, Y if at least the confidence threshold, then the raw scores
   ```
   `SESSIONS.TXT` lists each session, and `STATE.TXT` lets the board resume after a power cut.

`<name>_diag.bin` is a diagnostic version for checking a board. The card's files, the clock, power cuts and
replaying a known window are explained in [audiomoth/README.md](audiomoth/README.md).

**Not yet tested on a real board.** The firmware has passed every test on the simulated device and on an emulated
processor, but it has not run on an AudioMoth yet. Do a first test with a board on your desk before you deploy:
see [audiomoth/docs/FIRST_BOARD_TEST.md](audiomoth/docs/FIRST_BOARD_TEST.md).

## Measured so far

| run | data | test macro-F1, call windows (original → device version) |
|---|---|---|
| `tests/synthetic` | 3 synthetic tonal "species" + background, trained from scratch | 1.000 → 1.000; firmware built and tested |

**Limits:**
- No real-species training data has been measured yet.
- No long field recordings from an AudioMoth have been used yet. The selection of moments in long files is tested
  on synthetic data only (`tests/test_training.py`).
- False reports per hour need hours of background recordings from the deployment sites.
- The accuracy of the shipped bird model in `audiomoth/docs/VERIFICATION.md` §9 was measured on a previous
  version, made from different calibration audio, and has not been re-measured.

## Licences and credits

- **wren-framework** (`wren/`, `wrennet/`, the tests, tools and documentation): MIT licence; see `LICENSE`.
- **AudioMoth firmware** (`audiomoth/`): MIT licence, Open Acoustic Devices; see `audiomoth/LICENSE`.
- **Bird calibration recordings** (`examples/birds/calibration/`): 3 s excerpts of xeno-canto recordings under CC0,
  CC BY or CC BY-SA. Each clip keeps its licence. The recordist, licence and source of every clip are listed in
  `ATTRIBUTION.csv`.
- **WrenNet** model code and the bird model: `wrennet/`.

## For developers

How the pipeline works, what it checks, what is pinned, the tests and the folder layout:
[docs/PIPELINE.md](docs/PIPELINE.md). Changing a dependency: [envs/README.md](envs/README.md). Feature
specifications and plans: `specs/`.
