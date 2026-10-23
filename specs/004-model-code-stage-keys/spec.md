# Feature Specification: Stages re-run when the model code changes

**Feature Branch**: `004-model-code-stage-keys`

**Created**: 2026-10-23

**Status**: Draft

**Input**: User description: "4 — the export stage's key does not include the WrenNet model code (wrennet/), so changing
the model code does not re-run export by itself and a stale converted model is reused. Every stage whose output
depends on the model code must re-run when it changes."

## Context

The pipeline skips a stage when its inputs have not changed (constitution, Pipeline & Quality Gates). Each stage's key
lists what it depends on. Measured on the `002-clean-wrennet-audiomoth` code:

| Stage | Loads the model? | Key covers wren's model loader | Key covers WrenNet's code (`wrennet/`) |
|-------|------------------|-------------------------------|----------------------------------------|
| train | yes | yes | yes |
| evaluate | yes | through train's key | through train's key |
| export | yes (reference, backbone export, spectrograms) | yes | **no** |
| package | yes (tail weights, test reference) | **no** | **no** |
| firmware | no (uses package's output) | through package's key | through package's key |

So for a project that converts a given model (no train stage), a change to the model code leaves export and package
"up to date" and their old outputs are reused: the firmware would no longer match the code that claims to have made it.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — A change to the model code re-runs every stage that depends on it (Priority: P1)

A developer edits the WrenNet architecture or wren's model loader and runs `python -m wren run` on an existing project
without `--force`. Every stage whose output comes from the model re-runs; the others are skipped.

**Why this priority**: A stale output that looks up to date breaks constitution IV and V: the report would claim
checks on outputs the current code did not produce.

**Independent Test**: For a convert project and a training project, compute each stage's key, change one byte in a
`wrennet/` source file or in the model loader, and compare: the keys of export and package (and, for training
projects, train and evaluate) change; prepare's does not.

**Acceptance Scenarios**:

1. **Given** a convert project that ran to the end, **When** a `wrennet/` source file changes and the pipeline runs
   again, **Then** export, package and firmware re-run.
2. **Given** the same project, **When** wren's model loader changes, **Then** export, package and firmware re-run.
3. **Given** a training project, **When** the model code changes, **Then** train, evaluate, export, package and firmware
   re-run, and prepare does not.
4. **Given** no change at all, **When** the pipeline runs again, **Then** every stage is skipped, as today.

### Edge Cases

- A change outside the model's source files (README, bytecode caches) must not re-run anything: the key covers source
  files only, as the existing folder hash already does for `audiomoth/`.
- The bird checkpoint lives under `wrennet/`: its own key entry stays by content, so moving or renaming it does not
  matter, only its bytes.
- Outputs do not change: only the keys do, so a run after this feature re-runs the affected stages once, and the bird
  reproduction test (which forces every stage) is unaffected.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The export stage's key MUST change when any source file of the WrenNet model code changes.
- **FR-002**: The package stage's key MUST change when the WrenNet model code or wren's model loader changes, because
  package computes the tail weights and the test reference from the model.
- **FR-003**: Keys MUST NOT change for edits that cannot change outputs: documentation files in `wrennet/` and caches.
- **FR-004**: Stages that do not load the model (prepare) MUST NOT change their keys.
- **FR-005**: The stage-key tests MUST cover FR-001 to FR-004 for a convert project, and the model-code part of a
  training project's train key.
- **FR-006**: No output file of any stage changes; the bird reproduction test passes unchanged.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After a one-byte change to a model source file, a second run of a convert project re-runs export,
  package and firmware (3 of 3 affected stages) and skips none of them.
- **SC-002**: After a change to `wrennet/README.md` only, every stage is skipped.
- **SC-003**: The stage-key tests and the bird reproduction test pass in CI.

## Assumptions

- Feature 002 is merged first: it makes `wrennet/` the model package and is the code this feature changes.
- Hashing the model's source files costs well under a second, as the `audiomoth/` hash in the package key does today.
- Constitution V is unaffected: keys decide whether a stage runs, not what it produces.
