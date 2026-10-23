# Feature Specification: Clean up wrennet/ and audiomoth/

**Feature Branch**: `002-clean-wrennet-audiomoth`

**Created**: 2026-10-23

**Status**: Draft

**Input**: User description: "Clean up wrennet/ and audiomoth/ (the follow-up recorded at the end of
specs/001-reproducible-from-zero/tasks.md). wrennet/ becomes only what the pipeline uses: an importable WrenNet model
package (architecture + learned spectral front-end) and the bird checkpoint; the unused shims and WrenNet's original
training/distillation/dataset code, configs, soft labels and old conversion experiments are removed (they stay in the
WrenNet repository and in this repository's git history). The firmware's bird scripts no longer need a WRENNET_ROOT
argument. audiomoth/ drops files the pipeline and its documentation do not use (datasheets PDF, snippets, unused
tools), after checking tools/board. Every check of feature 001 must keep passing: CI, the byte-exact bird reproduction
in Docker, parity checks, firmware tests."

## Clarifications

### Session 2026-10-23
 The
questions below were answered by Claude from the repository and the 001 follow-up notes; each answer is marked for the
maintainer's review in the pull request.

- Q: WrenNet's original training and distillation code (tasks.md 001: "keep it as `wrennet/training/`, or drop it now
  that `wren` trains on its own") → A: Drop it. Nothing in the pipeline, the tests or the documentation runs it; it
  depends on packages the locked environments do not install (PyTorch Lightning, Hydra), so a kept copy would be
  dead code that cannot run. It stays in the WrenNet repository (`sciapponi/WrenNet` at 47c7059) and in this
  repository's history (commit 7a60637). *Maintainer review.*
- Q: Name and place of the model package and the checkpoint? → A: The folder `wrennet/` itself becomes the importable
  package `wrennet` (`wrennet/models/`, `wrennet/spectral/`), found from the repository root with no environment
  variable; the bird checkpoint moves to `wrennet/checkpoints/birds.pt` (same bytes).
- Q: May the model code be edited while it moves (trimmed of unused classes, unused imports removed)? → A: No. The
  architecture and front-end source files move with their content unchanged, so the computation is the same by
  construction; only the package's `__init__.py` files change. Trimming the model code, and dropping the Python
  packages it imports but does not use (`torchsummary`, `omegaconf`) from the locked environments, are separate work.
- Q: `audiomoth/tools/board/` (the 001 note: "check first")? → A: Keep. `docs/FIRST_BOARD_TEST.md` uses all four of
  its scripts, and `firmware/app/main_app.c` points to one. For the same reason `tools/energy_parser.py`
  (FIRST_BOARD_TEST step 5) and `tools/memcount.c` (memory profile in README and VERIFICATION) stay.
- Q: Which audiomoth files go? → A: Those that nothing in the repository runs or names: the datasheet PDF (replaced by
  a link to the published datasheet), the unreferenced screenshot `docs/images/image.png`, `firmware/snippets/`,
  `tools/bird_frontend_v1_reference.c.txt`, and the unreferenced developer scripts `firmware/bird/host_test.c`,
  `firmware/bird/map_sizes.py`, `firmware/bird/sweep_tail.py`, `tests/sim/check_wav.py`. Also the EFM32 device
  headers of the 63 microcontroller variants the AudioMoth does not have (19 MB of the vendored library; the board is
  an EFM32WG380F256).

## User Scenarios & Testing *(mandatory)*

### User Story 1 — The bird model and every check still work after the clean-up (Priority: P1)

A maintainer removes the unused code from `wrennet/` and `audiomoth/`. Everything feature 001 guarantees still holds:
the shipped bird package is reproduced byte for byte, the firmware binaries match in the Docker image, the synthetic
project trains and converts end to end, and all parity checks and firmware tests pass.

**Why this priority**: A clean-up that breaks the reproduction or a firmware test is worse than no clean-up.

**Independent Test**: Run the bird reproduction test (natively and in CI's Docker job), the unit checks and the
synthetic end-to-end project on the branch; all pass with the same outputs as on `main`.

**Acceptance Scenarios**:

1. **Given** the cleaned repository, **When** the bird reproduction test runs in the Docker image, **Then** the
   package files and both firmware binaries are byte-identical to the committed references.
2. **Given** the cleaned repository on a native install, **When** the bird reproduction test runs, **Then** the package
   files are identical to the committed reference, as on `main` before the clean-up.
3. **Given** the cleaned repository, **When** CI runs on the pull request, **Then** the docker, native-linux and
   native-macos jobs pass.

---

### User Story 2 — A developer finds only code that is used (Priority: P1)

A conservation-technology developer opens `wrennet/` to understand the model the pipeline trains. They find the
architecture, the learned front-end and the bird checkpoint, with a short README that says what each is and where the
original training code lives. Nothing else.

**Why this priority**: Today 143 files, among them dozens of configurations of a different training system, sit next to the
6 files the pipeline uses; a reader cannot tell which code is live.

**Independent Test**: List `wrennet/`: every file is imported by the pipeline, is the checkpoint, or is its README.

**Acceptance Scenarios**:

1. **Given** the cleaned `wrennet/`, **When** the developer lists it, **Then** it holds only the model package, the
   checkpoint and a README.
2. **Given** the pipeline, **When** it loads the model, **Then** it imports `wrennet` from the repository with no
   environment variable or path argument.

---

### User Story 3 — Firmware tools without a model path argument (Priority: P2)

A developer runs one of the firmware's model scripts (tail weights, reference vectors, front-end check, simulator log
check). The script finds the model code in the repository itself; the developer passes the checkpoint and data, not a
path to a WrenNet checkout.

**Why this priority**: The `WRENNET_ROOT` argument pointed at a separate repository's `tools/` folder, which no longer
exists here; the scripts only worked when launched by the pipeline.

**Independent Test**: The pipeline's package stage runs the scripts without the argument; the usage lines and the
README commands show no `WRENNET_ROOT`.

**Acceptance Scenarios**:

1. **Given** the cleaned firmware scripts, **When** the package stage runs them, **Then** it passes no model root and
   the outputs are byte-identical.
2. **Given** a developer with the torch environment, **When** they run `verify_log.py CKPT WORKDIR` as the firmware
   README shows, **Then** the script loads the model from the repository.

---

### User Story 4 — A smaller audiomoth/ (Priority: P3)

A user clones the repository. The firmware folder no longer carries a 2.8 MB PDF, an unused screenshot, dead snippets
and headers for microcontrollers the AudioMoth does not use.

**Why this priority**: Smaller clone and less to read, but no change to behaviour.

**Independent Test**: `git ls-files audiomoth` lists none of the removed files; the firmware and its tests build and
pass.

**Acceptance Scenarios**:

1. **Given** the cleaned `audiomoth/`, **When** the firmware, simulator, QEMU and file-system tests build and run,
   **Then** they pass.
2. **Given** the board-test procedure (`docs/FIRST_BOARD_TEST.md`), **When** a user follows it, **Then** every tool it
   names exists.

### Edge Cases

- A stage key or report hashes `wrennet/` and `audiomoth/`: their hashes change, so cached stages re-run once. This is
  expected and must not change any output byte.
- A run directory from before the clean-up (`runs/<name>/`) is reused: the changed hashes make the affected stages
  re-run instead of reusing stale outputs.
- `examples/birds/project.yaml` names the checkpoint path: it moves with the file.
- A user-written `project.yaml` that names `wrennet/best_distillation_model.pt` as `checkpoint:` fails with the usual
  plain "file not found" message from the project check; the README shows the new path.
- Documentation (README, PIPELINE.md, firmware README, model README, VERIFICATION.md) names removed files or old
  paths: every such mention is updated or removed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `wrennet/` MUST contain only the importable model package `wrennet` (the WrenNet architecture and its
  learned spectral front-end), the bird checkpoint, and a README stating what they are, their origin
  (`sciapponi/WrenNet` at 47c7059) and where the removed training code can be found.
- **FR-002**: The architecture and front-end source files MUST keep their content unchanged when they move; only the
  package `__init__.py` files may change.
- **FR-003**: The bird checkpoint MUST move to `wrennet/checkpoints/birds.pt` with identical bytes; every reference
  (pipeline defaults, example project, tests, documentation) MUST follow.
- **FR-004**: The pipeline MUST import the model package from the repository with no environment variable; the
  `WRENNET_ROOT` variable and the stand-in loader's path tricks for it are removed.
- **FR-005**: The firmware's model scripts (`gen_bird_tail.py`, `e2e_make_ref.py`, `check_frontend.py`,
  `tests/sim/verify_log.py`) MUST NOT take a `WRENNET_ROOT` argument; they MUST load the model from the repository,
  both when the pipeline runs them and when a developer runs them directly with the torch environment.
- **FR-006**: The files listed in Clarifications (audiomoth) MUST be removed, and the datasheet MUST be replaced by a
  link to the published AudioMoth datasheet.
- **FR-007**: `audiomoth/tools/board/`, `tools/energy_parser.py`, `tools/memcount.c` and every file a build script, a
  test script or a document names MUST stay.
- **FR-008**: Every output the pipeline compares byte for byte (the bird package files and, in Docker, the firmware
  binaries) MUST be unchanged; `tests/expected/` and `audiomoth/firmware/bird/model/` MUST NOT change except for path
  mentions in the model README.
- **FR-009**: All checks of feature 001 MUST pass: unit checks, synthetic end to end, bird reproduction, firmware tests,
  and the three CI jobs.
- **FR-010**: Documentation MUST name no removed file and no old path (`best_distillation_model.pt`,
  `birds_distillation_edge`, `WRENNET_ROOT`), except the feature folders `specs/001-…` (history) and `specs/002-…`
  (which describes the change).

### Key Entities

- **WrenNet model package**: the fixed architecture (constitution III) and its learned front-end, imported by the
  pipeline's torch side and by the firmware's model scripts.
- **Bird checkpoint**: the trained weights of the shipped 9-class bird model; input of the bird reproduction and the
  starting point of `init: wrennet` training.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The bird reproduction test reports "identical" for all 7 package files natively, and for the package and
  both firmware binaries in the Docker CI job.
- **SC-002**: The three CI jobs pass on the pull request.
- **SC-003**: `wrennet/` goes from 143 tracked files to at most 12, every one of them imported, loaded or a README.
- **SC-004**: `audiomoth/` loses at least 20 MB of tracked files (from about 30 MB), with no build, test or documented
  procedure losing a file it needs.
- **SC-005**: A search of the repository (outside the `specs/` feature folders) finds no `WRENNET_ROOT`,
  `birds_distillation_edge` or `best_distillation_model.pt`.

## Assumptions

- The WrenNet repository (`sciapponi/WrenNet`) stays available as the home of the original training and distillation
  code; this repository's git history keeps a copy too.
- The locked environments do not change in this feature, so the Docker image's dependency layers and the
  byte-reproducibility of feature 001 are unaffected.
- Python module paths and file locations of the model code do not enter the exported model files (the ONNX graph names
  come from the model's attribute names). The byte-exact reproduction test verifies this.
- The datasheet link points to Open Acoustic Devices' published AudioMoth Dev datasheet; the PDF stays in git history.
