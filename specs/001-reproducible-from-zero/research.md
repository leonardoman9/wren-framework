# Research: Reproducible from Zero

Facts observed in the repository on 2026-10-22, and the decisions they lead to. Each entry gives the Decision,
the Rationale and the Alternatives considered.

## R1. ARM compiler

- **Observed**: `firmware/app/build_app.sh` compiles with
  `$TFLM/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi-*`, which TFLM's
  `arm_gcc_download.sh` downloads at commit e0021ca. That is Arm GNU Toolchain 14.3.Rel1, MD5-checked, with
  builds for Linux x86_64, Linux aarch64 and darwin-arm64. The system `arm-none-eabi-gcc` (Homebrew 15.2) is
  not used. `docs/VERIFICATION.md §11` shows that 10.3-2021.10 gives the same size (−100 B) but not the same
  bytes.
- **Decision**: Pin GCC 14.3.1 through TFLM's download script; no separate toolchain step. Setup verifies
  that `gcc_embedded/bin/arm-none-eabi-gcc --version` reports 14.3.1. (The spec was updated: FR-008, during
  planning.)
- **Rationale**: This is the compiler the shipped firmware and its tests were verified with. It is already
  pinned and checksum-checked upstream, and it runs natively on Apple Silicon.
- **Alternatives**: GNU Arm 10.3-2021.10 was rejected: no Apple Silicon build, and it would need a full
  re-verification. Homebrew or apt `arm-none-eabi-gcc` was rejected: unpinned.

## R2. Python version and bootstrap

- **Observed**: `requirements-torch.txt` pins numpy 2.4.6, which requires Python ≥3.11. `cli.py` looks for
  `python3.11`. `cli.py` imports `wren.config` (PyYAML) and the stages at module level, so
  `python -m wren setup` fails on a bare interpreter.
- **Decision**: Python 3.11 only. Setup moves to `wren/setup.py` (standard library only), and `cli.py` imports
  `config`, `runner` and the stages inside `run()`. If the running interpreter is not 3.11, setup looks for
  `python3.11` on PATH (or `--python`) for the environments, and otherwise stops with a plain message.
- **Rationale**: This is the smallest change that satisfies FR-001 and the clarification.
- **Alternatives**: A shell `setup.sh` was rejected in clarification 5. Supporting 3.10 would require
  downgrading numpy, which risks the byte-exact reproduction.

## R3. Python dependency locking (FR-007)

- **Observed**: The two requirements files pin top-level packages only. Transitive dependencies (sympy,
  networkx, filelock, protobuf users, h5py deps, …) float, so a fresh install six months from now can differ.
- **Decision**: Keep the current files as `envs/requirements-{torch,tf}.in` (intent). Generate full locks with
  hashes per platform: `envs/lock/{torch,tf}-{linux-x86_64,macos-arm64}.txt`, made with
  `uv pip compile --python-version 3.11 --python-platform <p> --generate-hashes`. Users and Docker install with
  plain `pip install --require-hashes --no-deps -r <lock>`, so uv is only needed to regenerate the locks. On
  Linux, torch and torchaudio come from the CPU wheel index (`download.pytorch.org/whl/cpu`).
- **Rationale**: Exact and verifiable installs. The CPU-only wheels avoid about 3 GB of CUDA packages in Docker
  and CI, and the pipeline's deterministic stages run on the CPU.
- **Alternatives**: `pip freeze` from one machine was rejected because it does not cover other platforms.
  Poetry or pdm were rejected as a new user-facing tool.
- **To verify in tasks**: the byte-exact bird reproduction still passes with the CPU torch wheels on Linux.

## R4. Docker image (FR-003, FR-004, clarifications 2–3)

- **Observed**: `docker/Dockerfile` uses `python:3.11.9-slim-bookworm` by tag, unpinned apt packages, no
  `.dockerignore` (so `COPY . .` copies `runs/` and `.venv-*`), and builds TFLM. It has never been run end to
  end.
- **Decision**:
  - `FROM --platform=linux/amd64 python:3.11.9-slim-bookworm@sha256:<digest>`.
  - apt sources point to `snapshot.debian.org` at a fixed timestamp.
  - Pip installs from the Linux locks.
  - TFLM is fetched and built inside the image (GCC 14.3.1 x86_64).
  - A `.dockerignore` is added.
  - A `wren docker` helper is not needed; the README shows the `docker build` and `docker run` lines.
- **Rationale**: Byte-identity "on any host" requires the same CPU architecture inside the container. TF
  conversion and the compiler differ between amd64 and arm64. The digest and the Debian snapshot make
  rebuilds at different times equivalent without a published image.
- **Risk**: On Apple Silicon, `linux/amd64` runs emulated (Docker Desktop with Rosetta). TF 2.16 x86 wheels
  need AVX, and Rosetta's AVX support depends on the macOS version.
  - Task: measure on the developer's Mac.
  - Fallback 1: if a `linux/arm64` image gives identical bytes, allow both.
  - Fallback 2: otherwise document the minimum macOS version.

## R5. Calibration bundle (FR-013, FR-014, FR-015; clarification 1 as revised)

- **Observed**:
  - The shipped package was calibrated on 400 BioDCASE WAVs, and its test windows came from 120 more.
  - `pipe_vec.cc` in the firmware embeds one 3 s BioDCASE window (`g_pcm[96000]`).
  - `examples/biodcase/*`, `tests/other_classes/project.yaml` and `examples/birds/project.yaml` point to
    BioDCASE on the developer's disk.
  - `make_specs.py` (seed 0) and `e2e_make_ref.py` (seed 7) select files by sorted glob plus a fixed-seed
    shuffle. Their globs are `*.wav` and `*.mp3`.
  - Audio file names do not appear in any compared output.
- **Decision**:
  - BioDCASE is removed everywhere.
  - The bundle is `examples/birds/calibration/` with these files:
    - `audio/*.flac`: at least 400 clips, 3 s each, mono, PCM_16, at 32 kHz (the model rate, so the
      pipeline's resample is the identity), cut from xeno-canto recordings of the model's 8 species;
    - `SHA256SUMS`;
    - `ATTRIBUTION.csv`, with one row per clip: `clip`, `xc_id`, `species`, `recordist`, `licence`,
      `licence_url`, `source_url`, `start_s`;
    - `LICENSE.md`, which says that each clip keeps its original Creative Commons licence (CC0 / CC-BY /
      CC-BY-SA, as listed in `ATTRIBUTION.csv`), that clips were cut, resampled and re-encoded, and that the
      CC-BY-SA clips are redistributed under their BY-SA licence;
    - `README.md` (source, selection rule, how the clips were cut).
  - The source is the xeno-canto API v3 (`https://xeno-canto.org/api/3/recordings`, maintainer key in
    `XC_API_KEY`; the query syntax is `gen:<Genus> sp:<species>`, verified 2026-10-22). Recordings are queried
    per species.
  - Measured on 2026-10-22: of the 12,552 recordings of the 8 model species, only 82 are CC0/BY/BY-SA (40 at
    quality A/B, about 23 min, half of it one species).
  - Species order: the 8 model species first (every free recording, quality A–C), then the other 62 species
    of the project's 70-species list (a constant in the builder), until the bundle is full. Calibration only
    needs representative bird audio and background. The builder keeps only those whose `lic` field is CC0,
    CC-BY or CC-BY-SA. It downloads only the selected recordings. The 140k-file Drive
    copy is not used: it is 120 GB and lacks licence metadata.
  - `examples/birds/project.yaml` uses the existing keys `calibration.audio: [calibration/audio]` and
    `test.audio: [calibration/audio]`. No new config key and no manifest are needed. Selection stays the
    existing deterministic sort plus fixed-seed shuffle over a folder that never changes.
  - `.flac` is added to the globs in `wren/stages/export.py:audio_files`, `wren/torchside/make_specs.py` and
    `audiomoth/firmware/bird/e2e_make_ref.py`.
  - The bundle is built once by `tools/make_calibration_bundle.py`, a maintainer tool. Per species it picks
    recordings in xeno-canto ID order with a fixed seed and up to `--per-file` (default 2) non-overlapping 3 s
    windows from each, chosen by energy, so that the windows contain calls. It decodes, downmixes,
    resamples to 32 kHz and converts to int16 (rint and clip), then writes the FLAC files, `SHA256SUMS` and
    `ATTRIBUTION.csv`. After that, the committed bundle is the reference, not the tool.
- **Rationale**:
  - The clips are real audio of the model's species, and their licences permit redistribution with
    attribution.
  - NC licences are excluded so the repository carries no non-commercial-only data. ND licences are
    excluded because cutting a clip is a derivative work.
  - Storing at 32 kHz PCM_16 removes resampling from the reproduced path.
  - FLAC is lossless and about half the size of WAV.

- **Consequence (FR-015)**: The bird package is regenerated once, with the same checkpoint and new int8
  ranges and test vector. It is committed to `audiomoth/firmware/bird/model/` as the new reference (R6). The accuracy figures in
  `audiomoth/docs/VERIFICATION.md` §9 are marked as measured on the previous package; re-measuring them is
  follow-up work outside this feature.
- **Alternatives**:
  - BioDCASE in any form was rejected by the maintainer.
  - The maintainer's own recordings were the first choice, but none were available: the Drive folder is a
    xeno-canto dataset.
  - NC clips were rejected because the repository would then carry non-commercial-only data.
  - Synthetic audio was rejected because its int8 ranges do not represent birds. WAV was rejected because it is about twice the size.
- **Size**: 400 × about 100 KB FLAC ≈ 40 MB.

## R6. Firmware and WrenNet inside this repository (no submodules)

- **Observed**:
  - The firmware lives in the `third_party/audiomoth-fw` submodule (`leonardoman9/audiomoth-nn-deployment`):
    363 tracked files, about 31 MB. Most of it is `vendor/audiomoth` (Open Acoustic Devices, MIT; licence at
    the repo root).
  - The superproject records b9b50a1. The working tree is at cfa7ced, which exists only on
    `fix/crash-window-test` and fast-forwards from b9b50a1.
  - The BioDCASE-derived files are `firmware/bird/model/test/pipe_vec.cc` and `test/specs_one.npz`.
  - The WrenNet submodule (`sciapponi/WrenNet`, the maintainer's own work) has 1,428 tracked files. The
    pipeline uses only a small part of it:
    - `birds_distillation_edge/` (65 files, 700 KB): the model (`models/`) and the spectral front-end;
    - the shims `models.py`, `modules.py` and `differentiable_spec_torch.py`;
    - `best_distillation_model.pt` (328 KB).
  - wren reaches WrenNet through `WRENNET_ROOT` (`wren/torchside/model.py`, `runner.py`), through
    `stages/data.py` (checkpoint path) and through the firmware scripts (`WRENNET_ROOT` argument).
  - The rest of WrenNet:
    - original training and distillation code: `distillation/`, `datasets/`, `dataset_transfer/`,
      `config/`, `train*.py`, `extract_soft_labels.py` (about 500 KB);
    - generated outputs: `logs/`, `logs_backup/`, `benchmark/` (about 104 MB);
    - a backup checkpoint, a duplicate under `models/`, and `present.pdf`.
  - wren uses the firmware path in `config.py` (`firmware_repo`), `runner.py` (`submodule_commit` in stage
    keys) and the stages.
- **Decision** (clarification: firmware self-contained, copied without history):
  - Copy the tracked files of audiomoth-fw at cfa7ced (`git archive cfa7ced`) into `audiomoth/`.
  - Exclude `firmware/bird/model/test/pipe_vec.cc` and `test/specs_one.npz`. They are added only once
    regenerated from the calibration bundle (R5), so the BioDCASE versions never enter this repository's
    history.
  - Remove the submodule (`git submodule deinit` plus `git rm third_party/audiomoth-fw`, and the
    `.gitmodules` entry).
  - Keep `LICENSE` (MIT, Open Acoustic Devices) at `audiomoth/LICENSE`, and add a short `audiomoth/README.md`
    note saying the code was imported from audiomoth-nn-deployment at cfa7ced.
  - `config.firmware_repo` defaults to `ROOT/"audiomoth"`. The `.gitignore` ignores
    `audiomoth/third_party/tflite-micro`.
  - Stage keys use a content hash of `audiomoth/` (tracked files) instead of the submodule commit.
  - `audiomoth-nn-deployment` is left untouched (no merge, no push).
  - Copy WrenNet's code (all of it except generated outputs) and `best_distillation_model.pt` into
    `wrennet/`, without history (`git archive` at the recorded commit 47c7059, filtered). Excluded:
    `logs/`, `logs_backup/`, `benchmark/`, `best_distillation_model BACKUP 24.06.pt`, `models/` (duplicate
    checkpoint), `present.pdf`, and WrenNet's own Docker and `run_docker_*` files (superseded by wren's
    Dockerfile).
  - `WRENNET_ROOT` defaults to `ROOT/"wrennet"`. The checkpoint path in `stages/data.py`,
    `tests/test_training.py` and `examples/birds/project.yaml` follows. Stage keys hash `wrennet/` instead of
    using the submodule commit.
  - Remove the WrenNet submodule and `.gitmodules`. A plain `git clone` is then enough, and setup has no
    submodule check.
- **Rationale**: One repository holds everything the user needs: training, model, firmware and deployment.
  Copying without history keeps the BioDCASE window out, and leaving out the generated outputs keeps
  about 104 MB of logs out. Using the `audiomoth/` name avoids
  `firmware/firmware/...` nesting and the misleading `third_party` label.
- **Alternatives**: Merging into the external repo and keeping the submodule was rejected by the maintainer.
  `git subtree` with history was rejected because it would carry the BioDCASE window into this repository's
  history.

## R7. CI layout (FR-005, FR-006, SC-003)

- **Decision**: `.github/workflows/ci.yml`, triggered on push to `main` and on pull requests, with three jobs:
  1. `docker` (ubuntu-latest): `docker buildx build` with the GHA layer cache, then inside the container:
     `tests/test_training.py`, the synthetic end-to-end run and `tests/test_birds_reproduction.sh` (byte-exact
     package and firmware hash).
  2. `native-linux` (ubuntu-24.04, x86_64): `python -m wren setup --tflm`, then the synthetic end-to-end run
     (parity and gates).
  3. `native-macos` (macos-14, arm64): the same as `native-linux`.
  - The pip cache and the TFLM build directory are cached by key (lock hash; TFLM commit plus patch hash).
  - Synthetic data comes from `tests/make_synthetic_dataset.py` (no audio committed), with epochs reduced for
    CI.
- **Rationale**: Clarification 4. Caching keeps each job under 20 minutes (TFLM build about 5–10 min cold).
- **Alternatives**: A single Docker job was rejected in clarification 4.

## R8. Byte-exact firmware for the bird model (US3, FR-010)

- **Observed**: `test_birds_reproduction.sh` compares 7 package files and stops at `--stages export,package`.
  No firmware `.bin` is committed.
- **Decision**: Extend the test to `export,package,firmware`. Compare `runs/birds/firmware/birds.bin` and
  `birds_diag.bin` sha256 with `tests/expected/birds_firmware.sha256`, recorded from the first green Docker
  run. The package reference is the regenerated package of R5, produced inside the Docker image. The package comparison stays. The `.bin` comparison runs only inside Docker. Natively, the script
  reports the hash as information.
- **Rationale**: Clarification 2. It covers "the firmware binary is byte-for-byte identical" without
  committing binaries.

## R9. Training tolerance (Principle V)

- **Decision**: Training is not compared byte for byte. The synthetic project is accepted when it passes the
  evaluate→export gate (macro-F1 > 1/N + 0.1) and all later parity checks. CI uses CPU and a fixed seed. The
  README states that GPU or cross-platform training gives different weights, judged by the gates only.
- **Rationale**: The shipped bird model starts from a fixed checkpoint (no training), so its reproduction is
  unaffected.

## R10. Disk, RAM and system prerequisites (FR-011, FR-012)

- **Decision**: Setup checks these before anything is downloaded:
  - Python 3.11, `git`, `make`, a C compiler (`cc`), `wget` (TFLM's GCC download uses it; it is missing on
    stock macOS), `mtools` and `qemu-system-arm` (firmware tests).
  - At least 15 GB free disk (the spec's figure; replaced by measured numbers after the first CI runs).
  - RAM: 8 GB documented, not checked.
  - Each missing tool gets a one-line install hint for apt or Homebrew.
- **Rationale**: Fail early with clear messages (spec edge cases).

## R11. macOS x86_64

- **Observed**: PyTorch has published no macOS x86_64 wheels since 2.2, so torch 2.14.1 cannot be installed
  there.
- **Decision**: On macOS x86_64 (and any platform other than Linux x86_64 or macOS arm64), setup stops before
  creating any environment and directs the user to the Docker path with one plain sentence. Docker on Intel Macs is the supported route.
- **Rationale**: This is the honest reading of the clarification. A half-working native install would be
  worse.

## R12. Replacing the scratchpad symlinks (FR-002)

- **Decision**: If `.venv-torch` or `.venv-tf` is a symlink, setup removes the link (never its target) and
  creates a real venv. `--check` reports symlinked environments as an error.
- **Rationale**: Every pipeline module already resolves `ROOT/.venv-*`. Only the setup has to change.
