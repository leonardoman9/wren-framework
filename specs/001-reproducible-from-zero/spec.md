# Feature Specification: Reproducible from Zero

**Feature Branch**: `001-reproducible-from-zero`

**Created**: 2026-10-22

**Status**: Draft

**Input**: User description: "Renderlo riproducibile da zero" — make the wren-framework pipeline fully reproducible: anyone cloning the repo on a clean machine must be able to run the entire audio → model → firmware pipeline without manual setup steps, undocumented dependencies, or symlinks to session-specific paths.

## Clarifications

### Session 2026-10-22

- Q: On a clean machine, where should the bird-model reproduction get the exact calibration audio it needs? → A: Commit a fixed calibration bundle to the repo, verified by checksum. (Source superseded during planning — see the BioDCASE entry below.)
- Q: Must firmware be byte-identical on native (non-Docker) installs too, or only on the Docker path? → A: Byte-identical only on the Docker path (on any host); native installs must complete the pipeline and pass all parity checks, but their bytes may differ.
- Q: Should the Docker image be published ready-made to a registry, or does everyone build it from the Dockerfile? → A: No published image; users and CI each build it locally from the Dockerfile.
- Q: On which platforms must the native (non-Docker) install really work, and be verified automatically in CI? → A: Linux x86_64 and macOS ARM are supported and tested in CI; macOS x86 is best-effort and untested. (Refined during planning: PyTorch publishes no macOS x86 builds, so on macOS x86 the setup stops immediately and directs the user to Docker.)
- Q: What exact command does a user type right after cloning to install everything? → A: `python -m wren setup` (setup module uses only the Python standard library, same pattern as `python -m wren run`).
- Q (during planning): Which ARM compiler is pinned — GNU Arm 10.3-2021.10 or the GCC 14.3.1 that TFLM downloads? → A: GCC 14.3.1 (Arm GNU Toolchain 14.3.Rel1), downloaded and MD5-checked by TFLM's own script at the pinned commit; it is what the firmware and its tests already use, and it has native Apple Silicon builds (no Rosetta). 10.3 was only verified to give the same flash size, not the same bytes.
- Q (during planning): Which Python versions are supported? → A: Python 3.11 only (the pinned numpy 2.4.6 requires ≥3.11).
- Q (during planning): May BioDCASE audio be used or redistributed anywhere? → A: No. BioDCASE is dropped from the project entirely (calibration, test vectors, examples, tests, CI, docs). The firmware test vector that currently embeds a BioDCASE window is regenerated before the firmware enters this repository.
- Q (during planning): Which audio replaces BioDCASE for calibrating and testing the bird model? → A: (First answer: the maintainer's own recordings, CC-BY 4.0; superseded because the provided Drive folder turned out to be a third-party xeno-canto dataset.) Final: xeno-canto recordings fetched through the xeno-canto API, restricted to free licences (CC0, CC-BY, CC-BY-SA; no NC, no ND), committed as 3 s clips with per-clip attribution.
- Q (during planning): Only 82 of 12,552 xeno-canto recordings of the 8 model species are CC0/CC-BY/CC-BY-SA (about 23 min at quality A/B, very unbalanced). How is the bundle completed? → A: All freely licensed recordings of the 8 species first, then freely licensed recordings of other European species (the other 62 species of the project's species list) until the bundle is full; licences stay CC0/CC-BY/CC-BY-SA. Calibration needs representative bird audio, not the model's species.
- Q (during implementation): The README and the repository were reviewed on 2026-10-22: the README carries ML jargon, an outdated reproducibility paragraph and no device/deploy section; the repository tracks caches, dead files of the imported firmware repo and a maintainer tool among the tests. Fix them here or later? → A: Here, as FR-019 (README for biologists, technical details in `docs/PIPELINE.md`) and FR-020 (repository hygiene). The deeper restructuring of `wrennet/` and `audiomoth/` is a separate follow-up feature.
- Q (during planning): Should the firmware keep coming from the separate `audiomoth-nn-deployment` repository (git submodule)? → A: No. The firmware becomes part of this repository as plain files (a copy of commit cfa7ced, without its git history), and the submodule is removed.
- Q (during planning): Should WrenNet (model code and checkpoint) stay a submodule? → A: No. It is the maintainer's own work: its code and the bird checkpoint are copied into this repository as well, so wren-framework is one self-contained package for training, firmware and deployment, with no submodules.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Clone and run the pipeline on a clean machine (Priority: P1)

A researcher clones `wren-framework` on a fresh macOS or Linux machine. They have audio folders organised by class and want to produce a working AudioMoth firmware binary. They run a single setup command, write a `project.yaml`, and run `python -m wren run project.yaml`. Every stage — prepare, train, evaluate, export, package, firmware, report — completes without errors.

**Why this priority**: This is the entire point of the project. If a new user cannot go from zero to a firmware binary, nothing else matters.

**Independent Test**: Clone the repo on a clean machine (or a fresh Docker container), run `python -m wren setup --tflm && python -m wren run examples/birds/project.yaml` and verify the firmware binary is produced. (Until the calibration bundle of User Story 3 exists, the synthetic project `tests/synthetic/project.yaml` stands in for `examples/birds`.)

**Acceptance Scenarios**:

1. **Given** a clean machine with Python 3.11 and Docker installed, **When** the user runs `python -m wren setup` (or `docker build`), **Then** all dependencies (PyTorch, TensorFlow/ai_edge_litert, onnx2tf, ARM toolchain) are installed in isolated environments with pinned versions.
2. **Given** a working environment, **When** the user runs the pipeline on the example birds project, **Then** every stage succeeds; in the Docker image the outputs are the same as the reference (byte-for-byte where deterministic, within tolerance where not), and on a native install every parity check passes.
3. **Given** a clean machine without Docker, **When** the user runs `python -m wren setup`, **Then** the native setup succeeds on Linux x86_64 and macOS ARM (macOS 14 or later), with clear errors if a system dependency is missing; on macOS x86 it stops before downloading anything and directs the user to the Docker path.

---

### User Story 2 — CI runs the full pipeline on every push (Priority: P1)

A contributor pushes a commit to `wren-framework`. GitHub Actions runs the pipeline end-to-end on a synthetic mini-dataset (2–3 tonal classes + noise, a few files, the fewest epochs that still pass the evaluate→export quality gate) and reports pass/fail. The CI workflow uses the same Docker image or setup script as the user.

**Why this priority**: Without CI the pipeline silently breaks. The synthetic dataset makes CI fast (minutes, not hours) and avoids shipping real audio data.

**Independent Test**: Push a commit and verify the CI job passes — the firmware binary is built, host tests pass, and QEMU pipeline parity passes.

**Acceptance Scenarios**:

1. **Given** a push to `main` or a pull request, **When** CI runs, **Then** the synthetic end-to-end pipeline completes in under 20 minutes.
2. **Given** the CI environment (Ubuntu runner), **When** the pipeline runs, **Then** the Docker job uses the same Dockerfile and dependency versions as a local Docker build, and its firmware binary is byte-identical to a local run in the Docker image on the same commit.
3. **Given** a breaking change (e.g. a wrong TFLM patch), **When** CI runs, **Then** the relevant stage fails with a clear error message, and the job exits non-zero.

---

### User Story 3 — Reproduce the shipped bird model (Priority: P2)

A researcher wants to verify the 8-bird + non-bird model that the project ships. They run the pipeline on the bird checkpoint project (`examples/birds/project.yaml` with `checkpoint:` pointing to the existing `.pt` file and calibration/test audio from the committed calibration bundle). The exported TFLite, generated C code, and firmware binary are byte-for-byte identical to the ones in the repository.

Because the calibration audio changes from BioDCASE to freely licensed xeno-canto recordings, the shipped bird package is regenerated once (same float weights, new int8 ranges and test vector) and becomes the new committed reference. The accuracy figures measured on the previous package are marked as such; re-measuring accuracy on labelled audio is follow-up work outside this feature.

**Why this priority**: Reproducibility of the shipped artefacts is the credibility baseline. If the pipeline cannot reproduce what it already ships, its outputs for new data are suspect.

**Independent Test**: Run `python -m wren run examples/birds/project.yaml` and diff every output against the committed artefacts.

**Acceptance Scenarios**:

1. **Given** the bird checkpoint, the calibration bundle and the Docker image, **When** the pipeline runs, **Then** the TFLite, C tables, and firmware binary are byte-for-byte identical to the committed versions.
2. **Given** the Docker image running on a different host (macOS or Linux), **When** the pipeline runs on the same checkpoint, **Then** the outputs are still byte-identical (deterministic stages) or within documented tolerance (training).
3. **Given** a native (non-Docker) install, **When** the pipeline runs on the same checkpoint, **Then** it completes and passes every parity check (front-end + core, ONNX vs PyTorch, int8 vs float, device pipeline vs PyTorch), but its output bytes are not required to match the committed artefacts.

---

### User Story 4 — Bring-your-own data on a new machine (Priority: P2)

An ecologist with audio recordings of 4 frog species + background noise clones the repo on a university Linux workstation, runs setup, writes a minimal `project.yaml`, and gets a firmware binary and a report with accuracy numbers — without any knowledge of PyTorch, TensorFlow, ONNX, or ARM toolchains.

**Why this priority**: This is the real-world use case the framework is built for. If it only works on the developer's laptop, it's not a framework.

**Independent Test**: On a clean Linux machine, place audio folders, write a 10-line YAML, run the pipeline, and check that the report and `.bin` file are produced.

**Acceptance Scenarios**:

1. **Given** audio folders with ≥5 files per class at any sample rate, **When** the user runs the pipeline, **Then** prepare resamples to 32 kHz, train reaches the quality gate, and firmware is built.
2. **Given** a `project.yaml` with only `data:` and `negative:`, **When** the user runs the pipeline, **Then** all other parameters use documented defaults and the pipeline completes.

---

### Edge Cases

- What happens when Docker is not installed and the user runs `python -m wren setup`? → On Linux x86_64 and macOS ARM the native setup runs. On any other platform (macOS x86, other OS or CPU) it stops before downloading anything with a plain message pointing to the Docker path.
- What happens when the user has a different Python version (3.10, 3.12)? → The setup checks the version and fails early with a message stating that Python 3.11 is required.
- What happens when disk space is insufficient? → The setup checks for ~6 GB of free space (measured need ~4 GB plus margin) before downloading anything and stops with a plain message stating how much is needed and how much is free.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The project MUST provide a single setup command (`python -m wren setup`) that installs all dependencies from pinned versions into isolated virtual environments. The setup command MUST run with a bare Python 3.11 interpreter (standard library only), before any dependency is installed.
- **FR-002**: The setup MUST NOT depend on symlinks to session-specific or machine-specific paths (the current `.venv-torch` and `.venv-tf` symlinks to scratchpad directories must be replaced).
- **FR-003**: The project MUST provide a Dockerfile that builds a self-contained image with all dependencies (PyTorch, TensorFlow/ai_edge_litert, onnx2tf, ARM GCC toolchain, TFLM) at pinned versions. No pre-built image is published: users and CI build it locally. So that builds made at different times yield an equivalent image, the Dockerfile MUST pin the base image by digest, pin system packages, and verify every downloaded artefact (toolchain archive, TFLM source) by checksum.
- **FR-004**: The Docker image MUST be sufficient to run the entire pipeline from audio folders to firmware binary without any host dependencies beyond Docker itself.
- **FR-005**: The project MUST include a GitHub Actions CI workflow that runs the full pipeline on a synthetic mini-dataset and reports pass/fail, in three jobs: inside the Docker image (Linux), native setup on Linux x86_64, and native setup on macOS ARM. The native jobs check the parity checks and quality gates, not byte-identity.
- **FR-006**: The CI workflow MUST build its image from the same Dockerfile the user builds (layer caching allowed), so that CI and local runs are equivalent.
- **FR-007**: All Python dependencies MUST be pinned to exact versions in lock files or requirements files, not ranges.
- **FR-008**: The ARM GCC toolchain version MUST be pinned to Arm GNU Toolchain 14.3.Rel1 (GCC 14.3.1), downloaded and checksum-verified by TFLM's download script at the pinned TFLM commit — not whatever Homebrew or apt installs.
- **FR-009**: The TFLM source MUST be fetched at a pinned commit (e0021ca) and patched with the project's patches, via a reproducible script (`audiomoth/scripts/setup_tflm.sh`).
- **FR-010**: The `examples/birds` project MUST pass a byte-exact reproduction test (`tests/test_birds_reproduction.sh`) as part of CI, run inside the Docker image. Byte-exactness is guaranteed only on the Docker path; native installs are held to the parity checks instead.
- **FR-011**: The project MUST document the supported platforms — Docker on any host; native on Linux x86_64 and macOS ARM, macOS 14 or later (tested in CI; the pinned PyTorch has macOS wheels for 14+ only); macOS x86 and other platforms via Docker only — plus the Python version and disk/RAM requirements.
- **FR-012**: The setup MUST check its prerequisites (platform, Python version, disk space, required system tools) before downloading anything and fail early with clear, non-technical messages.
- **FR-013**: The int8 calibration and test audio for `examples/birds` MUST be a calibration bundle of 3 s clips cut from xeno-canto recordings — first every usable recording of the model's 8 species, then other European bird species (a fixed list of 62 species) until the bundle is full — fetched through the xeno-canto API and restricted to recordings licensed CC0, CC-BY or CC-BY-SA (NC and ND licences excluded); it is committed to the repository with a checksum list and a per-clip attribution file (xeno-canto ID, recordist, licence and its URL, species, source URL, start offset), and each clip keeps its original licence. `examples/birds/project.yaml` MUST reference it by a repo-relative path, and no example or test project may reference absolute or machine-specific paths.
- **FR-014**: No BioDCASE audio, path or derived data may appear in the project: the `examples/biodcase` projects and BioDCASE paths in test projects are removed, README/docs results that depend on BioDCASE are removed, and the firmware test vectors (`firmware/bird/model/test/pipe_vec.cc`, which embeds a 3 s BioDCASE window, and `specs_one.npz`) are regenerated from the calibration bundle; the BioDCASE versions MUST never be committed to this repository.
- **FR-015**: The shipped bird package (`audiomoth/firmware/bird/model/`) MUST be regenerated with the calibration bundle (same checkpoint) and committed as the new byte-exact reference; every parity check and firmware test MUST pass on it. The accuracy figures in `audiomoth/docs/VERIFICATION.md` §9 MUST be marked as measured on the previous package; re-measuring them is out of scope for this feature.
- **FR-016**: The AudioMoth firmware (application, bird model runtime, vendored AudioMoth libraries, TFLM scripts, firmware tests) MUST live in this repository under `audiomoth/`, copied from `audiomoth-nn-deployment` at commit cfa7ced without its git history; the `third_party/audiomoth-fw` submodule is removed, and the original MIT licence and Open Acoustic Devices attribution are kept.
- **FR-017**: The WrenNet model code (`birds_distillation_edge/` package, original training and distillation code, configs, the compatibility modules `models.py`, `modules.py` and `differentiable_spec_torch.py`) and the bird checkpoint `best_distillation_model.pt` MUST live in this repository under `wrennet/`, copied from the WrenNet repository without its git history; generated outputs (`logs/`, `logs_backup/`, `benchmark/` results), backup checkpoints and the presentation PDF are not copied. After this, the repository MUST have no git submodules, and a plain `git clone` (without `--recursive`) MUST be enough.
- **FR-018**: Every run report MUST record how it was produced (constitution VI): Docker reference path or native install, the platform, the lock files used and the source versions. Only Docker runs are declared byte-reproducible; native runs are declared "verified by parity checks".
- **FR-019**: The top-level `README.md` MUST be written for biologists (constitution I): what the framework does, setup, supported platforms, Docker, what is reproducible, `project.yaml` with every parameter and default, measured results, how to put the firmware on the device (flashing, SD card `CONFIG.TXT`, reading the log), and licences/credits (wren-framework code, the AudioMoth firmware under MIT / Open Acoustic Devices, the calibration clips under their CC licences). It MUST NOT contain ML or toolchain jargon (tensors, quantisation, ONNX, TFLM operator names) outside a single pointer to `docs/PIPELINE.md`, which holds the technical stage-by-stage description. Every statement MUST match the current repository (paths, hashes, measured setup time and disk need).
- **FR-020**: The repository MUST NOT track caches, build output or dead files: `.pytest_cache/` is git-ignored; the imported firmware repository's own CI (`audiomoth/.github/`) and its Raspberry Pi benchmarks (`audiomoth/benchmarks/`, including an empty BirdNET placeholder) are removed; maintainer tools live in `tools/` (not `tests/`); test files have descriptive names (`tests/test_step3.py` → `tests/test_training.py`); finished planning documents (`docs/PLAN_STEP3.md`) are removed (git history keeps them). References to moved or renamed files are updated everywhere (README, CI, Dockerfile, specs' contracts).

### Key Entities

- **Virtual environment (torch)**: PyTorch, torchaudio, ONNX, onnxruntime, onnxsim — used by prepare, train, evaluate, export (ONNX export, spectrograms) and package stages.
- **Virtual environment (tf)**: TensorFlow, onnx2tf, ai_edge_litert — used by export (ONNX → TensorFlow → int8 TFLite), package (device-pipeline parity) and the firmware tests.
- **ARM toolchain**: `arm-none-eabi-gcc` 14.3.1 (Arm GNU Toolchain 14.3.Rel1, fetched by TFLM at the pinned commit) — used by firmware stage. Native builds exist for Linux x86_64, Linux aarch64 and macOS ARM.
- **WrenNet**: the model architecture code, its original training/distillation code and the bird checkpoint under `wrennet/` (plain files in this repository).
- **TFLM library**: TensorFlow Lite for Microcontrollers at commit e0021ca + patches — used by firmware stage.
- **Firmware**: the AudioMoth application and its tests under `audiomoth/` (plain files in this repository, MIT) — built by the firmware stage.
- **Calibration bundle**: 3 s clips from freely licensed xeno-canto recordings (CC0 / CC-BY / CC-BY-SA) — the model's 8 species first, completed with other European bird species — used to quantise the bird model (calibration + int8-vs-float evaluation) and to build its test windows; committed with checksums and per-clip attribution. Changing it changes the int8 model bytes and the test vector.
- **Docker image**: Contains all of the above, plus the repo itself.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user on a clean machine (macOS or Linux) can go from `git clone` to a firmware `.bin` file in under 30 minutes of hands-on time (excluding download and training time).
- **SC-002**: The `examples/birds` byte-exact reproduction test passes on every CI run.
- **SC-003**: CI completes the synthetic end-to-end pipeline in under 20 minutes on a standard GitHub Actions runner.
- **SC-004**: The Docker image produces identical firmware binaries on macOS and Linux hosts for the same inputs and pinned dependencies. Native installs are not required to be byte-identical, but MUST pass every parity check and quality gate.
- **SC-005**: Zero undocumented manual steps between cloning and running the pipeline — every step is in `python -m wren setup` or the Dockerfile.

## Assumptions

- Users have Python 3.11 installed, or are willing to install it. Native macOS users run macOS 14 or later (older versions use Docker).
- Users have at least 6 GB of free disk space for the environments and toolchain (measured: ~4 GB), plus room for their own runs; the Docker path needs about 10 GB more for the image.
- Docker is available for the fully reproducible (byte-identical) path; native setup on macOS and Linux must complete the pipeline and pass all parity checks, but is not byte-identical.
- This narrows constitution principle V ("byte-identical firmware on any supported platform (macOS, Linux, Docker)"); a constitution amendment MUST accompany this feature.
- The Arm GNU Toolchain 14.3.Rel1 archives remain downloadable from developer.arm.com at the URLs pinned in TFLM's download script.
- GitHub Actions runners have enough resources (disk, RAM) for the firmware build and QEMU tests.
- The TFLM commit e0021ca remains accessible in the TensorFlow repository.
- Freely licensed recordings are about 1% of xeno-canto: the 8 model species alone give 82 recordings (about 23 min at quality A/B, measured 2026-10-22), so the bundle is completed with other European species; across the 70-species list this is expected to exceed the ~400 + 120 clips needed. Only the selected recordings are downloaded, never the full 140k-file Drive dataset. The API key is needed only to build the bundle, never by users or CI.
