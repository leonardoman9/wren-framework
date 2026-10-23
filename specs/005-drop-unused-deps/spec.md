# Feature Specification: Drop unused packages from the torch environment

**Feature Branch**: `005-drop-unused-deps`

**Created**: 2026-10-23

**Status**: Draft

**Input**: User description: "5 — the WrenNet model code imports torchsummary and omegaconf without using them; drop the
imports and the two packages from the locked torch environment, keeping the Docker bird reproduction byte-exact."

## Context

Measured on the `002-clean-wrennet-audiomoth` code:

- `wrennet/models/legacy_modules.py` imports `torchsummary.summary` (never called) and `omegaconf.DictConfig` (used only
  as a type annotation of two constructor arguments, never at run time).
- `envs/requirements-torch.in` pins `torchsummary`, `omegaconf` (which brings `antlr4-python3-runtime`) and `tqdm`.
  Nothing in `wren/`, `wrennet/`, `tests/`, `tools/` or the firmware's scripts imports `tqdm` either: it was there for
  WrenNet's removed training code.

Every user installs these packages (`python -m wren setup`, the Docker image) for nothing.

## Clarifications

### Session 2026-10-23

- Q: Also `tqdm`, which the request did not name? → A: Yes. It is unused for the same reason (WrenNet's removed
  training code), and removing it goes through the same lock change and the same verification.
- Q: Feature 002 kept the model files byte-identical (its FR-002). May this feature edit them? → A: Yes, only the import
  lines and the two type annotations; no executable statement changes. The bird reproduction proves the computation is
  unchanged.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Install only what the pipeline uses (Priority: P1)

A researcher runs `python -m wren setup` (or builds the Docker image). The torch environment no longer downloads and
installs `torchsummary`, `omegaconf`, its parser runtime and `tqdm`; everything works as before.

**Why this priority**: Fewer packages to download, audit and keep compatible; the environment says what the pipeline
really needs.

**Independent Test**: After setup, the four packages are absent from the torch environment, and the unit checks, the
synthetic end-to-end project and the bird reproduction pass.

**Acceptance Scenarios**:

1. **Given** the new locks, **When** the user runs setup on Linux x86_64 or macOS arm64, **Then** the torch environment
   has none of the four packages and setup's own checks pass.
2. **Given** the Docker image built from the new locks, **When** the bird reproduction test runs, **Then** the package
   files and both firmware binaries are byte-identical to the committed reference.
3. **Given** an existing install made with the old locks, **When** the user runs setup again, **Then** setup notices that
   the locks changed and brings the environment in line, as it does for any lock change.

### Edge Cases

- A package that one of the four brought in is also needed by something else (e.g. `PyYAML`): it stays, because it is
  pinned at the top level or required by another kept package; the lock tool decides, not a hand edit.
- The type annotations: removing `DictConfig` must not change how the model is built from its plain-dictionary
  configuration (`wren/torchside/model.py` passes a dict).
- The Docker image's dependency layer is rebuilt once; its build cache in CI is invalidated once.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The model code MUST NOT import `torchsummary` or `omegaconf`; the two annotations that name `DictConfig`
  MUST no longer need it, and no executable statement of the model code changes.
- **FR-002**: `envs/requirements-torch.in` MUST NOT pin `torchsummary`, `omegaconf` or `tqdm`, and all four lock files
  MUST be regenerated with the project's lock tool (`envs/lock.sh`), not edited by hand.
- **FR-003**: Packages that only the removed ones required MUST disappear from the locks (e.g. the `omegaconf` parser
  runtime); every other locked version MUST stay the same.
- **FR-004**: The bird reproduction MUST stay byte-exact in the Docker image (package and firmware); the unit checks,
  the synthetic end-to-end project and the three CI jobs MUST pass.
- **FR-005**: Comments and documentation that list the torch environment's packages MUST match the new `.in` file.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The torch environment installs 4 fewer packages (`torchsummary`, `omegaconf`, `antlr4-python3-runtime`,
  `tqdm`), and no locked version of any other package changes.
- **SC-002**: The Docker bird reproduction reports "identical" for all package files and both firmware binaries.
- **SC-003**: The three CI jobs pass.

## Assumptions

- No user code outside this repository relies on these packages being present in `.venv-torch`.
- `envs/lock.sh` and its tool are available to the developer making the change, as `envs/README.md` describes.
- Feature 002 is merged first (the model code lives in `wrennet/models/`).
