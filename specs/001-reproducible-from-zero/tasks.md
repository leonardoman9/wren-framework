---
description: "Task list for 001-reproducible-from-zero"
---

# Tasks: Reproducible from Zero

**Input**: Design documents from `specs/001-reproducible-from-zero/`

**Prerequisites**: plan.md, spec.md, research.md (R1–R12), data-model.md, contracts/ (cli-setup, calibration-bundle,
docker-image, ci-workflow), quickstart.md, constitution 2.0.0

**Tests**: The spec asks for tests as deliverables: FR-010 (the reproduction test), FR-012 and the setup contract
(setup checks), and FR-013/FR-014 (example hygiene). They appear below as ordinary tasks.

**Organization**: Tasks are grouped by user story. Phases 1–2 consolidate the repository (no submodules, no
BioDCASE, new paths) and block every story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: US1–US4 from spec.md
- All paths are relative to the repository root `/Users/leonardomannini/Progetti/wren-framework`

---

## Phase 1: Setup (one self-contained repository)

**Purpose**: Bring WrenNet and the firmware into the repository as plain files, then remove both submodules
(FR-016, FR-017, research R6).

- [X] T001 Check that nothing local would be lost before removing the submodules:
  - run `git -C third_party/WrenNet status --short` and `git -C third_party/WrenNet log --oneline @{u}..HEAD`;
  - run `git -C third_party/audiomoth-fw status --short` and confirm cfa7ced exists on
    `origin/fix/crash-window-test`;
  - stop and report if there are uncommitted or unpushed changes. Do not push anything.
- [X] T002 Import WrenNet into `wrennet/` without history:
  - `mkdir wrennet && git -C third_party/WrenNet archive 47c7059 | tar -x -C wrennet`;
  - delete from `wrennet/`: `logs/`, `logs_backup/`, `benchmark/`, `best_distillation_model BACKUP 24.06.pt`,
    `models/` (duplicate checkpoint), `present.pdf`, `Dockerfile`, `Dockerfile.benchmark`, `docker-compose.yml`,
    `run_docker_*.sh` and `.dockerignore`;
  - keep `birds_distillation_edge/`, `config/`, `datasets/`, `distillation/`, `dataset_transfer/`, `tests/`, the
    top-level `*.py`, `README.md`, `requirements.txt` and `best_distillation_model.pt`;
  - add a first line to `wrennet/README.md`: "Imported from the WrenNet repository at 47c7059, without history;
    generated logs and benchmark results left out".
- [X] T003 [P] Import the firmware into `audiomoth/` without history:
  - `mkdir audiomoth && git -C third_party/audiomoth-fw archive cfa7ced | tar -x -C audiomoth`;
  - delete `audiomoth/firmware/bird/model/test/pipe_vec.cc` and `audiomoth/firmware/bird/model/test/specs_one.npz`.
    They contain BioDCASE audio and must never be committed (FR-014); T037 regenerates them;
  - keep `audiomoth/LICENSE` (MIT, Open Acoustic Devices);
  - add a first paragraph to `audiomoth/README.md`: "Imported from leonardoman9/audiomoth-nn-deployment at
    cfa7ced, without history".
- [X] T004 If `third_party/audiomoth-fw/third_party/tflite-micro` exists, move it to
  `audiomoth/third_party/tflite-micro` so that TFLM need not be rebuilt locally. It is git-ignored by
  `audiomoth/.gitignore`; verify with `git check-ignore`.
- [X] T005 Remove both submodules (depends on T001–T004):
  - `git submodule deinit -f third_party/WrenNet third_party/audiomoth-fw`;
  - `git rm -f third_party/WrenNet third_party/audiomoth-fw`;
  - delete `.gitmodules` and `.git/modules/third_party`;
  - confirm that `git submodule status` prints nothing and that `third_party/` no longer exists.
- [X] T006 [P] Update `.gitignore` (keep `runs/`, `.venv-*`, `__pycache__/`, `*.pyc` and `.DS_Store`; add
  `audiomoth/third_party/tflite-micro/`) and create `.dockerignore` with `runs/`, `.venv-*`,
  `audiomoth/third_party/tflite-micro/`, `**/__pycache__`, `.git` and `.claude/`.
- [X] T007 Scan the imported trees: `grep -rIl -i -E "biodcase|/Users/|/private/tmp|/home/[a-z]" wrennet audiomoth`
  must return nothing. Fix or remove any hit; report non-trivial ones.

**Checkpoint**: `git clone` (without `--recursive`) would contain everything; `third_party/` is gone.

---

## Phase 2: Foundational (paths, bootstrap import, locks, BioDCASE removal)

**Purpose**: Make the existing code work on the new layout and prepare the pieces every story needs.

**⚠️ CRITICAL**: No user-story work can begin until this phase is complete.

- [X] T008 Point wren at the new folders:
  - in `wren/config.py`, set `firmware_repo` to `ROOT / "audiomoth"`; the `tflm` default then becomes
    `audiomoth/third_party/tflite-micro`;
  - in `wren/torchside/model.py`, set the `WRENNET_ROOT` default to `ROOT / "wrennet"` and update the
    docstring;
  - in `wren/runner.py`, set the `WRENNET_ROOT` env default to `ROOT / "wrennet"`;
  - in `wren/stages/package.py`, change the `ROOT / "third_party" / "WrenNet"` arguments (lines 44 and 61) to
    `ROOT / "wrennet"`.
- [X] T009 Replace the submodule commits in stage keys with content hashes:
  - in `wren/runner.py`, replace `submodule_commit(name)` with `tree_hash(path)`: a sha256 over the sorted
    relative paths and the contents of the files under `path`, skipping git-ignored build output
    (`audiomoth/third_party/tflite-micro`, `__pycache__`);
  - update the callers in `wren/stages/data.py` (WrenNet) and `wren/stages/package.py` (the
    `git -C fw rev-parse HEAD` in the key, lines 24–27).
- [X] T010 [P] Move the checkpoint paths to `wrennet/best_distillation_model.pt`:
  - `wren/stages/data.py` lines 41–43;
  - `tests/test_step3.py` line 55;
  - `examples/birds/project.yaml` (`checkpoint: ../../wrennet/best_distillation_model.pt`);
  - also `docker/Dockerfile`, `tests/test_birds_reproduction.sh` (`B=audiomoth/firmware/bird/model`) and the
    `README.md` layout section (line 141 and around). Finally `git grep -n third_party` must hit only
    `audiomoth/` internals.
- [X] T011 Make `python -m wren setup` stdlib-only (contract cli-setup.md, Preconditions):
  - in `wren/cli.py`, move the imports of `wren.config`, `wren.runner` and `wren.stages` inside `run()`;
  - check with `python3.11 -I -c "import sys; sys.argv=['wren','setup','--help']; import runpy; runpy.run_module('wren', run_name='__main__')"`
    on an interpreter without PyYAML.
- [X] T012 Python locks (FR-007, research R3; data-model "Lock file"):
  - `git mv envs/requirements-torch.txt envs/requirements-torch.in`, and the same for tf;
  - add `pytest==<current release, exact pin>` to `envs/requirements-torch.in` (T014, T020 and CI need it);
  - generate `envs/lock/{torch,tf}-{linux-x86_64,macos-arm64}.txt` with
    `uv pip compile envs/requirements-<env>.in --python-version 3.11 --python-platform <x86_64-unknown-linux-gnu|aarch64-apple-darwin> --generate-hashes`;
  - for `torch-linux-x86_64`, add `--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match`
    so that torch and torchaudio are CPU wheels;
  - start each lock with a comment holding the exact command and `sha256(<.in file>)`;
  - write `envs/README.md` with the regeneration commands.
- [X] T013 [P] Remove BioDCASE (FR-014):
  - delete `examples/biodcase/`;
  - remove the BioDCASE rows and paragraphs in `README.md` (around lines 49–53 and 139) and item 3 "Real data,
    new classes" in `docs/PLAN_STEP3.md` (around line 147);
  - leave `tests/other_classes/project.yaml` for T036 (it needs the calibration bundle).
- [X] T014 [P] Create `tests/test_examples.py` (pytest, torch env; data-model "Calibration bundle" →
  Validation). It fails if:
  - any `*.yaml` under `examples/` or `tests/` contains an absolute path or the string "biodcase"
    (case-insensitive);
  - `examples/birds/calibration/ATTRIBUTION.csv` and `audio/*.flac` do not match one to one (checked when the
    folder exists);
  - any `licence` is not CC0, CC-BY or CC-BY-SA ("only xeno-canto recordings licensed CC0, CC-BY or
    CC-BY-SA").

**Checkpoint**: `.venv-torch/bin/python tests/test_step3.py` passes on the new layout, and
`.venv-torch/bin/python -m wren run tests/synthetic/project.yaml` still completes.

---

## Phase 3: User Story 1 — Clone and run on a clean machine (Priority: P1) 🎯 MVP

**Goal**: `git clone` → `python -m wren setup --tflm` → `python -m wren run …` works natively on Linux x86_64
and macOS arm64, and through Docker anywhere.

**Independent Test**: On a fresh clone, natively and in the Docker image:
1. run `python3.11 -m wren setup --tflm`;
2. run `python3.11 -m wren setup --check` (exit 0);
3. run `python3.11 -m wren run tests/synthetic/project.yaml` after `tests/make_synthetic_dataset.py`.

The firmware `.bin` must be produced (quickstart §1–2). `examples/birds` joins this test after US3.

- [X] T015 [US1] Create `wren/setup.py` (stdlib only) with the precondition checks of contract cli-setup.md,
  in order. Use the message texts from the contract table verbatim, plain words with no jargon (Principle I).
  Exit with code 2 on the first failure:
  1. platform is Linux x86_64 or macOS arm64; otherwise point to Docker;
  2. Python 3.11: the current interpreter, else `python3.11` on PATH, else `--python`;
  3. at least 15 GB free in the repo's filesystem (`shutil.disk_usage`). The threshold comes from one constant,
     calibrated in T023. The `WREN_MIN_DISK_GB` environment variable may lower it, for CI runners only and
     documented in `envs/README.md`, not in the user README;
  4. tools `git`, `make`, `cc`, `wget`, `mtools`, `qemu-system-arm`, with one apt or brew hint line each;
  5. the lock for this platform exists and its header hash equals `sha256(.in)`.
- [X] T016 [US1] Environment creation in `wren/setup.py` (contract Actions 1; data-model "Environment"), for
  `.venv-torch` and `.venv-tf`:
  - if it is a symlink, `os.unlink` the link only and never its target;
  - if it is missing, or its `.wren-lock` content differs from the lock's sha256, create the venv with the
    3.11 interpreter, run `pip install --require-hashes --no-deps -r envs/lock/<env>-<platform>.txt` and
    write `.wren-lock`;
  - otherwise skip it.
- [X] T017 [US1] The `--tflm` step in `wren/setup.py` (contract Actions 2; data-model "Toolchain and TFLM"):
  - skip it if both `libtensorflow-microlite.a` files exist and `git -C audiomoth/third_party/tflite-micro rev-parse HEAD`
    starts with e0021ca;
  - otherwise run `sh audiomoth/scripts/setup_tflm.sh` and then `HOST=1 sh audiomoth/scripts/build_tflm.sh`;
  - afterwards, `…/downloads/gcc_embedded/bin/arm-none-eabi-gcc --version` must contain `14.3.1`;
  - write the output to `runs/_setup/logs/setup.log`; on failure exit 1 with a plain message naming that log.
- [X] T018 [US1] Output, `--check` and `--verbose` in `wren/setup.py` (contract Synopsis, Actions 3 and exit
  codes):
  - the final line is "Ready. Next: python -m wren run your-project.yaml";
  - `--check` changes nothing and prints one plain line per item ("Analysis tools: ready", "Firmware build
    tools: missing — run python -m wren setup --tflm"); it exits non-zero if anything is missing;
  - `--check --verbose` adds the component names and versions;
  - versions otherwise go only to `setup.log`.
- [X] T019 [US1] Wire up the CLI in `wren/cli.py`:
  - the `setup` subparser gets `--tflm`, `--check`, `--verbose` and `--python`, and dispatches to
    `wren.setup.main`;
  - delete the old `setup()` function;
  - update the module docstring.
- [X] T020 [P] [US1] Create `tests/test_setup.py` (pytest, stdlib plus monkeypatch) covering:
  - the Python-version message for 3.12;
  - macOS x86_64 refused with the Docker message;
  - the disk shortfall message showing the needed and free space;
  - a missing tool line with its install hint;
  - a stale lock header detected;
  - a symlinked `.venv-tf` replaced while its target is untouched;
  - a second run with matching `.wren-lock` doing no install.
- [X] T021 [US1] Rewrite `docker/Dockerfile` per contracts/docker-image.md:
  - base image `FROM --platform=linux/amd64 python:3.11.9-slim-bookworm@sha256:<digest>`; get the amd64
    digest from `docker buildx imagetools inspect python:3.11.9-slim-bookworm`;
  - apt sources set to `snapshot.debian.org/archive/debian/<fixed timestamp>` before `apt-get install` of
    build-essential, git, wget, make, mtools, qemu-system-arm and libsndfile1;
  - venvs from `envs/lock/*-linux-x86_64.txt` with `--require-hashes --no-deps`;
  - `COPY . .`, then `sh audiomoth/scripts/setup_tflm.sh && HOST=1 sh audiomoth/scripts/build_tflm.sh`;
  - `ENV WREN_PATH=docker`, `ARG WREN_IMAGE_ID` and `ENV WREN_IMAGE_ID=$WREN_IMAGE_ID`;
  - the entrypoint stays `.venv-torch/bin/python -m wren`;
  - update the usage comment at the top (no `--recursive`).
- [X] T022 [US1] Add a provenance block to the report in `wren/stages/report.py` (FR-018; data-model "Run
  provenance"; Principle VI). The fields are:
  - `path`: `docker` if `WREN_PATH=docker`, else `native`;
  - `image`: `WREN_IMAGE_ID`, or null;
  - `platform`;
  - `locks`: sha256 of the lock files used;
  - `sources`: the wren-framework commit plus a dirty flag, and `tree_hash` of `wrennet/` and `audiomoth/`;
  - `byte_reproducible`.
  The human-readable report says either "Docker reference path: byte-reproducible" or "native install:
  verified by parity checks, not byte-reproducible".
- [X] T023 [US1] Validate natively on this Mac (macOS arm64):
  - remove `.venv-torch`, `.venv-tf` and `audiomoth/third_party/tflite-micro` (after T004, move it aside
    instead of deleting it);
  - run `python3.11 -m wren setup --tflm`, then `--check`, then the synthetic project;
  - record the wall time and the disk used (`du -sh .venv-torch .venv-tf audiomoth/third_party/tflite-micro`);
  - set the threshold constant in `wren/setup.py` to the measured need plus a 30% margin, and update the 15 GB
    figure in spec.md (edge cases, assumptions), `contracts/cli-setup.md` and `README.md` if it changes
    (research R10).
- [X] T024 [US1] Validate Docker on this Mac:
  - `docker build --platform linux/amd64 -f docker/Dockerfile -t wren .`, then run the synthetic project in
    the container (quickstart §2 command pattern);
  - research R4 risk: if TensorFlow fails under amd64 emulation, record the error and keep a `linux/arm64`
    build of the same Dockerfile for local use. The byte comparison with amd64 happens in T042;
  - document the outcome in `README.md`.
- [X] T025 [P] [US1] Update `README.md` per FR-011 and SC-005:
  - "Setup" section: clone without `--recursive`, Python 3.11, the one-line apt/brew prerequisites,
    `python -m wren setup --tflm`, `--check`;
  - "Using Docker" section: the build and run lines from contracts/docker-image.md;
  - a supported-platforms table: Docker on any host; native Linux x86_64 and macOS arm64 (CI-tested); macOS
    x86 and other platforms via Docker only;
  - requirements: Python 3.11, the disk figure from T023, 8 GB RAM;
  - a short note that native outputs are verified by parity checks, while Docker outputs are
    byte-reproducible;
  - the training tolerance (constitution V; research R9): training is not compared byte for byte. CPU with a
    fixed seed (as in CI) repeats on the same platform; GPU or other platforms give different weights. A
    trained model is accepted by the quality gates (macro-F1 > 1/N + 0.1, int8 within 0.05 of float), not by
    its bytes.

**Checkpoint**: US1 Independent Test passes natively (macOS arm64) and in Docker.

---

## Phase 4: User Story 2 — CI on every push (Priority: P1)

**Goal**: `.github/workflows/ci.yml` with the `docker`, `native-linux` and `native-macos` jobs
(contracts/ci-workflow.md).

**Independent Test**: Open a PR. All three jobs are green, each in under 20 minutes with warm caches. A
deliberately broken commit fails with a plain `FAILED:` line and a non-zero exit (quickstart §4).

- [X] T026 [US2] Tune `tests/synthetic/project.yaml` for CI:
  - lower `training.epochs` to the smallest value that still passes the evaluate→export gate (macro-F1 > 1/N
    + 0.1) on CPU;
  - measure the end-to-end time inside Docker;
  - add a comment with the measured time and margin.
- [X] T027 [US2] Create `.github/workflows/ci.yml`, triggered on push to `main` and on `pull_request`, with
  a plain checkout. Its `docker` job (ubuntu-latest):
  - `docker/setup-buildx-action`, then `docker buildx build --platform linux/amd64 --load -t wren` with
    `cache-from/cache-to: type=gha` and `--build-arg WREN_IMAGE_ID=${{ github.sha }}`;
  - inside the image (`.venv-torch/bin/python` as the interpreter): `python tests/test_step3.py` (a plain
    script), `python -m pytest tests/test_setup.py tests/test_examples.py`, `python tests/make_synthetic_dataset.py`,
    `python -m wren run tests/synthetic/project.yaml`.
- [X] T028 [P] [US2] Add the `native-linux` job (ubuntu-24.04) to `.github/workflows/ci.yml`:
  - `apt-get install` the prerequisites from contract cli-setup.md check 4;
  - print `df -h .`; if the runner has less free space than the setup threshold but enough for the measured
    need from T023, set `WREN_MIN_DISK_GB` for this job;
  - `actions/setup-python` with 3.11;
  - `python -m wren setup --tflm`, `python -m wren setup --check`, then the synthetic run;
  - no byte comparison.
- [X] T029 [P] [US2] Add the `native-macos` job (macos-14) to `.github/workflows/ci.yml`: `brew install wget
  mtools qemu`, Python 3.11, then the same steps as `native-linux`, including the `df -h .` and
  `WREN_MIN_DISK_GB` handling.
- [X] T030 [US2] Caches and failure artefacts in `.github/workflows/ci.yml`:
  - `actions/cache` for pip, keyed on `hashFiles('envs/lock/**')`;
  - `actions/cache` for `audiomoth/third_party/tflite-micro`, keyed on the TFLM commit plus
    `hashFiles('audiomoth/firmware/patches/**', 'audiomoth/scripts/build_tflm.sh')`;
  - `actions/upload-artifact` of `runs/**/logs` with `if: failure()`.
- [ ] T031 [US2] Open a PR from the feature branch and get all three jobs green. Record the warm-cache
  durations (SC-003). Then, on a throwaway branch, break `audiomoth/firmware/patches/tflm-no-int4.patch`. CI
  must fail with a plain `FAILED:` line (US2 scenario 3). Delete the throwaway branch afterwards.

**Checkpoint**: CI green on the PR; a breaking change turns it red.

---

## Phase 5: User Story 3 — Reproduce the shipped bird model (Priority: P2)

**Goal**: The bird package is regenerated from a committed, freely licensed calibration bundle, and its
byte-exact reproduction (package and firmware `.bin`) passes in Docker on every CI run (FR-010, FR-013,
FR-015, SC-002, SC-004).

**Independent Test**: `docker run --rm --platform linux/amd64 -v "$PWD/runs:/wren/runs" --entrypoint tests/test_birds_reproduction.sh wren`
prints `PASS`, with all 7 package files identical and both firmware sha256 values matching (quickstart §2).

- [X] T032 [US3] Maintainer action: get a xeno-canto API key (xeno-canto account → API key) and export it
  as `XC_API_KEY` in the shell that runs T034. Never write it to a file.
- [X] T033 [US3] Create `tests/make_calibration_bundle.py` per contracts/calibration-bundle.md "Builder". It
  must:
  - query the xeno-canto API v3 with `query=gen:<Genus> sp:<species>` (syntax verified), over all pages, 1 s
    between requests;
  - take species in this order: first the 8 model species of `examples/birds/project.yaml` (not
    `non-bird`), then `FILL_SPECIES`, the other 62 names of the project's 70-species list (Accipiter gentilis,
    Aegithalos caudatus, Aegolius funereus, Ardea cinerea, Asio otus, Athene noctua, Buteo buteo, Caprimulgus
    europaeus, Carduelis carduelis, Chloris chloris, Coccothraustes coccothraustes, Columba palumbus, Corvus
    corax, Cuculus canorus, Curruca communis, Cyanistes caeruleus, Delichon urbicum, Dendrocopos major,
    Dryobates minor, Dryocopus martius, Emberiza cirlus, Erithacus rubecula, Falco tinnunculus, Ficedula
    hypoleuca, Fringilla coelebs, Garrulus glandarius, Hirundo rustica, Jynx torquilla, Lanius collurio, Milvus
    milvus, Motacilla alba, Motacilla cinerea, Muscicapa striata, Oriolus oriolus, Otus scops, Parus major,
    Passer montanus, Pernis apivorus, Phoenicurus ochruros, Phoenicurus phoenicurus, Phylloscopus bonelli,
    Phylloscopus collybita, Phylloscopus sibilatrix, Picus canus, Picus viridis, Poecile palustris, Prunella
    modularis, Pyrrhula pyrrhula, Regulus ignicapilla, Regulus regulus, Serinus serinus, Sitta europaea, Spinus
    spinus, Strix aluco, Sturnus vulgaris, Sylvia atricapilla, Troglodytes troglodytes, Turdus merula, Turdus
    philomelos, Turdus pilaris, Turdus viscivorus, Upupa epops);
  - keep only licences CC0, CC-BY and CC-BY-SA (`lic` URL), rejecting any `-nc` or `-nd`, and only quality
    A, B or C;
  - take every kept recording of the 8 model species first, then fill round-robin across `FILL_SPECIES` with
    `--seed`, and download only the selected files into a temporary directory;
  - decode, downmix, resample to 32 kHz with torchaudio, take the top-RMS non-overlapping 3 s windows (up to
    `--per-file`), and convert to int16 with `rint` and clipping;
  - write `audio/XC<id>_<ssss>.flac` (FLAC, PCM_16, mono, 32 kHz, 3.0 s each), `SHA256SUMS`,
    `ATTRIBUTION.csv` (`clip,xc_id,species,recordist,licence,licence_url,source_url,start_s`), `LICENSE.md`
    and `README.md`. The README states the species mix (model species compared with fill species);
  - read the API key from the environment only.
- [X] T034 [US3] Build the bundle:
  - `XC_API_KEY=… .venv-torch/bin/python tests/make_calibration_bundle.py examples/birds/calibration --clips 400 --test 120`;
  - check that there are at least 400 clips plus the test material, the size (target about 40 MB), and that
    `tests/test_examples.py` passes;
  - report how many clips come from the 8 model species and how many from fill species;
  - read `ATTRIBUTION.csv` once to confirm the licences. Never relax the licence filter.
- [X] T035 [P] [US3] Add `*.flac` to the audio globs:
  - `wren/stages/export.py` `audio_files`;
  - `wren/torchside/make_specs.py` (header: "copied from WrenNet; +flac");
  - `audiomoth/firmware/bird/e2e_make_ref.py`.
- [X] T036 [US3] Rewrite `examples/birds/project.yaml` so it uses only repo-relative paths:
  - `checkpoint: ../../wrennet/best_distillation_model.pt`;
  - `calibration: {audio: [calibration/audio], count: 300, eval: 100}`;
  - `test: {audio: [calibration/audio], windows: 120}`;
  - update the header comment, dropping the `audiomoth-nn-deployment` path;
  - in `tests/other_classes/project.yaml`, set `calibration.audio: [../../examples/birds/calibration/audio]`.
- [X] T037 [US3] Regenerate the reference package in Docker (FR-015):
  - rebuild the image, then run `docker run --rm --platform linux/amd64 -v "$PWD/runs:/wren/runs" wren run examples/birds/project.yaml --force`;
  - every parity check and the firmware tests must pass;
  - copy `runs/birds/package/{model.tflite→bird_backbone_int8.tflite, generated/bird_frontend_tables.h, generated/bird_tail_weights_i8.h, generated/tail_dims.h, generated/model_config.h, model_config.json, test/pipe_vec.cc, test/specs_one.npz}`
    into `audiomoth/firmware/bird/model/`;
  - confirm that `pipe_vec.cc` now comes from a bundle clip (its name is in `runs/birds/package_work/e2e_ref.npz`).
- [X] T038 [US3] Record `tests/expected/birds_firmware.sha256` in `sha256sum` format, with lines for
  `birds.bin` and `birds_diag.bin`, from `runs/birds/firmware/` of the T037 Docker run.
- [X] T039 [US3] Update `tests/test_birds_reproduction.sh`:
  - first run `(cd examples/birds/calibration && sha256sum -c --quiet SHA256SUMS)`, and stop with the failing
    clip's name;
  - run `--stages export,package,firmware --force`;
  - compare the 7 package files with `audiomoth/firmware/bird/model/` as now;
  - compare the `.bin` hashes with `tests/expected/birds_firmware.sha256` when `WREN_PATH=docker`, otherwise
    print them as information;
  - update the header comment: what is byte-exact, and on which path.
- [X] T040 [US3] Mark `audiomoth/docs/VERIFICATION.md` §9 as "measured on the previous package (calibrated on
  different audio); not re-measured for the current package". Update
  `audiomoth/firmware/bird/model/README.md` to name the calibration bundle as the source of the int8 ranges
  and the test vector.
- [X] T041 [US3] Add `tests/test_birds_reproduction.sh` to the `docker` job in `.github/workflows/ci.yml`
  (after the synthetic run).
- [ ] T042 [US3] Confirm SC-004: the reproduction test passes in the CI `docker` job (Linux host) and in
  Docker on this Mac (macOS host) with the same commit. If T024 found amd64 emulation unusable on this Mac,
  run the reproduction in the `linux/arm64` build instead and compare its bytes with the CI amd64 result:
  - if they are identical, allow both architectures (research R4, fallback 1);
  - otherwise record that SC-004 is verified on Linux hosts only, and say so in `README.md`.

**Checkpoint**: The bird reproduction passes in Docker locally and in CI; `examples/birds` now also works in
the US1 Independent Test.

---

## Phase 6: User Story 4 — Bring-your-own data (Priority: P2)

**Goal**: A minimal YAML (data folder + negative class) runs end to end with documented defaults
(constitution II, spec US4 scenario 2).

**Independent Test**: Generate the synthetic data, then run `python -m wren run tests/minimal/project.yaml`,
where the YAML contains only `data:` and `negative:`. The firmware and the report are produced.

- [X] T043 [US4] In `wren/config.py`, when `name` is missing, default it to the name of the YAML's parent
  folder, cleaned to `[A-Za-z0-9._-]`, instead of failing with "missing 'name'". Update the docstring.
- [X] T044 [US4] Create `tests/minimal/project.yaml` with only `data: {root: ../../runs/synth_data}` and
  `negative: background`, plus a comment explaining that it tests US4 scenario 2.
- [ ] T045 [US4] Run the minimal project natively and in Docker. All stages must complete with default
  parameters. If the default training is too long for CI, add a config-load check of `tests/minimal` to
  `tests/test_examples.py` instead of a full CI run, and note the measured time.
- [X] T046 [P] [US4] Check that `README.md` documents every `project.yaml` parameter and its default
  (constitution II), and add any that are missing.

**Checkpoint**: Minimal YAML works; defaults documented.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T047 Final hygiene scan over the whole repository:
  - `git grep -n -i -E "biodcase|third_party/(WrenNet|audiomoth-fw)|/Users/|/private/tmp|--recursive|wren setup\b"`;
  - fix every hit except historical notes in `specs/`;
  - confirm that `git ls-files | grep -E "pipe_vec.cc|specs_one.npz"` lists only the regenerated files.
- [ ] T048 Run quickstart.md scenarios 1–5 on this Mac (native and Docker) and record the results in the PR
  description. For scenario 1, follow only `README.md` from a fresh clone. Record every manual step and the
  hands-on time (excluding download, build and training waits) to check SC-001 (under 30 minutes) and SC-005
  (no undocumented steps).
- [X] T049 Remove the "Sync Impact Report" HTML comment from the top of `.specify/memory/constitution.md`
  before the final commit.
- [ ] T050 Re-run `/speckit-analyze` and resolve any remaining CRITICAL or HIGH findings.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: T001 first. T002, T003 and T006 can run in parallel. T004 comes after T003. T005
  comes after T001–T004. T007 comes after T002 and T003.
- **Phase 2 (Foundational)**: depends on Phase 1. T008 → T009. T010, T013 and T014 can run in parallel.
  T011 and T012 are independent of each other.
- **US1 (Phase 3)**: depends on Phase 2. T015 → T016 → T017 → T018 → T019. T020 can run in parallel after
  T015. T021 needs T012. T023 needs T019. T024 needs T021.
- **US2 (Phase 4)**: depends on US1 (T019, T021).
- **US3 (Phase 5)**: depends on US1 (Docker, T021) and Phase 2. T033 needs T032. T034 needs T033. T037
  needs T034–T036. T041 needs T027.
- **US4 (Phase 6)**: depends on Phase 2 only, so it can run alongside US1–US3.
- **Polish**: after all stories.

### Within Each Story

- Tests (T020, T014) are written alongside the code they check, and must pass before the story's checkpoint.
- The reference package (T037) is produced only inside Docker, never natively.

### Parallel Opportunities

- Phase 1: T002 ∥ T003 ∥ T006.
- Phase 2: T010 ∥ T013 ∥ T014, and T011 ∥ T012.
- US1: T020 ∥ T016–T018, and T025 at any time after T021.
- US2: T028 ∥ T029.
- US3: T035 ∥ T033/T034.
- US4: the whole phase runs in parallel with US1–US3.

---

## Parallel Example: User Story 1

```text
After T015:  T016 (environments)  ∥  T020 (tests/test_setup.py)
After T021:  T024 (Docker validation)  ∥  T025 (README)
```

## Parallel Example: User Story 3

```text
T033 (bundle builder)  ∥  T035 (flac globs)  ∥  T036 (examples/birds/project.yaml)
```

---

## Implementation Strategy

### MVP first (Phases 1–3)

1. Consolidate the repository (Phase 1) and fix the paths (Phase 2).
2. Native setup and Docker (US1).
3. **Stop and validate**: run the US1 Independent Test on this Mac, natively and in Docker.

### Then

4. CI (US2), which guards everything after it.
5. The calibration bundle and the regenerated bird reference (US3). This needs the xeno-canto API key
   (T032).
6. The minimal YAML (US4) can be slotted in at any point after Phase 2.
7. Polish and a final `/speckit-analyze`.

### Notes

- Nothing is pushed to `audiomoth-nn-deployment` or WrenNet. Both stay as they are, and this repository
  takes copies.
- Commit after each phase. The first commit that adds `audiomoth/` must not contain the BioDCASE
  `pipe_vec.cc` or `specs_one.npz` (T003).

---

## Follow-up (separate feature, after this one): clean up `wrennet/` and `audiomoth/`

Requested by the maintainer on 2026-10-22. It is deliberately out of scope here, so that the clean-up can be
verified by this feature's CI and byte-exact bird reproduction. Start it with `/speckit-specify`. Findings
so far:

- **wrennet/**:
  - The pipeline imports only `birds_distillation_edge/{__init__,models/{core,legacy_models,legacy_modules},spectral/differentiable_spec}.py`
    and `best_distillation_model.pt`. Nothing imports the shims `models.py`, `modules.py` or
    `differentiable_spec_torch.py`.
  - Everything else is WrenNet's original training and distillation code: `distillation/`, `datasets/`,
    `dataset_transfer/`, `config/`, `train*.py`, `extract_soft_labels.py`, `generate_no_birds_samples.py`,
    the soft-label JSONs, `tests/`, and the `birds_distillation_edge/{cli,data,distillation,losses,metrics,optim,utils,convert-and-quantize}`
    folders.
  - Decision needed: keep it as `wrennet/training/`, or drop it now that `wren` trains on its own. Candidate
    layout: an importable `wrennet` package (`wrennet/models`, `wrennet/spectral`) plus `wrennet/checkpoints/`.
- **audiomoth/**:
  - Used by the pipeline: `firmware/{app,bird,patches,audiomoth_tflm.ld}`, `scripts/`, `tests/{sim,qemu,fstest,schedule}`,
    `tools/gen_frontend_tables.c`, `vendor/audiomoth`, and `docs/{TFLM_BUILD,VERIFICATION,FIRST_BOARD_TEST}.md`.
  - Candidates for removal: `.github/` (the old repo's CI, dead inside a subfolder), `benchmarks/raspberry`,
    `docs/datasheets` (2.8 MB PDF), `firmware/snippets`, `tools/{board,energy_parser.py,memcount.c,bird_frontend_v1_reference.c.txt}`.
    Check `tools/board` first: it may be useful for deployment debugging.
- The `WRENNET_ROOT` argument of the firmware's bird scripts (`gen_bird_tail.py`, `e2e_make_ref.py`,
  `check_frontend.py`, `sweep_tail.py`, `tests/sim/verify_log.py`) can go once the model is an importable
  package.

---

## Phase 8: Convergence

- [X] T051 CRITICAL: rewrite `README.md` for biologists per Constitution I, FR-019 (contradicts):
  - drop the outdated "Reproducibility." paragraph (hand-made package, sha `6f77fa86…`) and link to "What is
    reproducible" instead;
  - replace the technical "What it does" table with a plain description of the stages;
  - fix the "firmware repository" wording (now `audiomoth/`) and the broken sentence in "Measured so far";
  - update "Layout" (`docker/`, `.github/`, `specs/`, `docs/`, `tools/`, `envs/` = `.in` + locks, the new tests;
    `make_specs.py` is no longer "unchanged");
  - keep a single pointer to `docs/PIPELINE.md`;
  - replace the setup time, download and disk estimates with the values measured in T023.
- [X] T052 CRITICAL: make the pipeline's console output plain-language per Constitution I (contradicts):
  - in `wren/runner.py`, `Runner.run` prints a plain label per step instead of the tool step name (map in the
    stages, e.g. `onnx2tf` → "converting the model for the device"), and `StepFailed` says what failed in plain
    words and where the log is, without the raw 15-line tool tail. The step name, command and output tail stay in
    `logs/<step>.log`;
  - in `wren/cli.py`, rephrase the "torch/tf environment missing" message the same way;
  - quality-gate messages keep naming the failing metric (constitution, Pipeline & Quality Gates).
- [X] T053 CRITICAL: enforce firmware verification per Constitution IV (contradicts). In `wren/stages/firmware.py`,
  when `firmware.tests` is false (`wren/config.py`), do not deliver `<name>.bin` as a flashable result: stop with a
  plain message, or write it only as `<name>_UNTESTED.bin` with `firmware.json` and `report.md` saying so. Document
  the behaviour in `README.md` (`firmware: tests:`).
- [X] T054 Create `docs/PIPELINE.md` per FR-019 (missing), holding the technical stage-by-stage description: the
  current "What it does" table, the parity checks, ONNX/TFLite/TFLM details and the two-environment split, linked
  from `README.md`.
- [X] T055 Add to `README.md` an "On the device" section per FR-019 (missing): flash `<name>.bin` with the AudioMoth
  Flash App, the SD card's `CONFIG.TXT`, what the log and LEDs mean, and the diagnostic build, linking to
  `audiomoth/README.md` for details. Also add a "Licences and credits" section: the wren-framework code, the
  AudioMoth firmware (MIT, Open Acoustic Devices, `audiomoth/LICENSE`), and the calibration clips (CC0/BY/BY-SA,
  `examples/birds/calibration/ATTRIBUTION.csv`).
- [X] T056 Restructure `report.md` in `wren/stages/report.py` per Constitution I (partial): first a plain-language
  summary (what was made, does it fit on the device, did every check pass, Docker or native), then the existing
  technical rows under a "Technical details" heading.
- [X] T057 Move `tests/make_calibration_bundle.py` to `tools/make_calibration_bundle.py` per FR-020 (missing), and
  update every reference:
  - `ROOT` in the script (now `parents[1]` of `tools/`, still the repo root);
  - the generated `examples/birds/calibration/README.md`;
  - `specs/001-reproducible-from-zero/{contracts/calibration-bundle.md,research.md,plan.md}`;
  - this file's T033/T034 text stays as history.
- [X] T058 Rename `tests/test_step3.py` to `tests/test_training.py` per FR-020 (missing), and update the references
  in `.github/workflows/ci.yml`, `README.md`, `specs/001-reproducible-from-zero/{quickstart.md,contracts/ci-workflow.md}`
  and the file's own usage line.
- [X] T059 Remove `audiomoth/.github/` (the old repository's CI) and `audiomoth/benchmarks/` (Raspberry Pi
  benchmarks, with an empty BirdNET placeholder) per FR-020 (missing). Update `audiomoth/README.md` and
  `audiomoth/.gitignore` where they mention them.
- [X] T060 Add `.pytest_cache/` to `.gitignore` and `.dockerignore` per FR-020 (missing).
- [X] T061 Remove `docs/PLAN_STEP3.md` per FR-020 (missing). Replace the reference in `wren/config.py` (the `data`
  field comment) with `README.md` ("project.yaml") and check `git grep PLAN_STEP3`.
- [X] T062 State the macOS 14 minimum per FR-011, US1/AC3 (partial). It is enforced by `wren/setup.py`
  (`platform_name`) and the macOS locks. Add it to spec.md FR-011, US1 scenario 3 and Assumptions, and to
  `contracts/cli-setup.md` check 1.
- [X] T063 Document in `specs/001-reproducible-from-zero/contracts/cli-setup.md` and `README.md` that
  `python -m wren run` re-runs itself inside `.venv-torch` when the typed Python lacks the pipeline's packages, per
  contracts/cli-setup.md (unrequested: implemented in `wren/cli.py` to keep constitution II's single command).
