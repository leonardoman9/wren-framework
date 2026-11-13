# Feature Specification: Fix the BirdNET target and measure whether it helps

**Feature Branch**: `008-birdnet-target-fix`

**Created**: 2026-10-23

**Status**: Draft

**Input**: User description: "Fix the BirdNET distillation target and measure it properly: on species windows the
target gives a large background share whenever BirdNET hears a species outside the project, contradicting the folder
label (for Periparus ater the target's top class is the folder's species on only 50% of the training windows). Change
the target so species windows never pull towards background (renormalise over the project's species, or skip BirdNET's
part when its top species is outside the project), then compare with and without BirdNET on the feature-006 test
bundle with several seeds, from scratch and from wrennet, with intervals, and also try a lower BirdNET weight."

## Context

Feature 003 built the target from BirdNET's confidences for the project's species plus a background share, the chance
that none of them is present. On a window of a species folder where BirdNET hears mostly a species outside the project
(another bird singing, or a confusion), that share is large, so half of the loss teaches "background" to a real call.
The report of `tests/birds_birdnet` shows it: the target's top class was the folder's species on 50–100 % of a
species' training windows (50 % for *Periparus ater*), while BirdNET named a species outside the project on up to 33 %.
The 003 comparison (74 test windows, one seed) could not tell whether BirdNET helps: the same run gave 0.59 and 0.79
on two computers. Feature 006 now provides a larger, freely licensed labelled set.

## Clarifications

### Session 2026-10-23

- Q: Which correction? → A: On a species window the target is renormalised over the project's species only (no
  background share), so BirdNET still says how sure it is and which other project species it hears; when BirdNET gives
  every project species less than 0.1 its part is skipped, as today. Background windows keep the background share.
  The experiment also runs the 003 target, to show what the correction changes.
- Q: Which data for the comparison, given that the 006 test bundle uses every freely licensed recording of the 8
  species (so no other free recordings are left to train on)? → A: Cross-validation by recording on the 006 bundle:
  the recordings are split into 4 folds (each recording, with all its windows, in one fold; species and non-bird
  sounds alike); each variant is trained on 3 folds and tested on the fourth, 4 times (4, not 5: the
  species with the fewest free recordings have 4, and every test fold must contain every class). No recording is ever in both
  training and test.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — BirdNET's opinion never contradicts a species label (Priority: P1)

A researcher turns on learning from BirdNET. On the windows of their species folders, BirdNET's opinion can make the
model more or less sure of the folder's species or spread some weight to the project's other species, but never
teaches that a real call is background.

**Why this priority**: The 003 target does exactly that on a fifth to a third of some species' windows; it is a
defect, not a tuning question.

**Independent Test**: Unit tests on the target: a species window where BirdNET hears only an outside species gives a
target with no background share; a background window keeps it.

**Acceptance Scenarios**:

1. **Given** a species window where BirdNET gives the project's species 0.3 and 0.05 and an outside species 0.9,
   **When** the target is built, **Then** it has no background share and its largest class is the project species at
   0.3.
2. **Given** a background window, **When** the target is built, **Then** it is as in feature 003.
3. **Given** the training report, **When** a project learns from BirdNET, **Then** "target's top class = folder" is
   reported per species and is now high wherever BirdNET hears the species at all.

---

### User Story 2 — A sound answer to "does BirdNET help?" (Priority: P1)

The maintainer runs one command that trains every variant several times on the cross-validation folds and writes a
comparison with intervals: with and without BirdNET, from scratch and from WrenNet, two BirdNET weights, the old and
the new target.

**Why this priority**: Turning BirdNET on should be recommended, or not, on evidence; today the evidence is noise.

**Independent Test**: The command produces a table whose intervals come from the folds and seeds, and the
documentation states the conclusion it supports, including "no difference" if that is the result.

**Acceptance Scenarios**:

1. **Given** the 006 bundle, **When** the comparison runs, **Then** each variant has a test macro-F1 and a
   false-detection rate with a 95 % interval over folds and seeds.
2. **Given** the results, **When** the README describes learning from BirdNET, **Then** it states what was measured,
   on what, and whether it helped, in plain words.

### Edge Cases

- BirdNET hears only species outside the project on a species window, all project species below 0.1: BirdNET's part
  is skipped (unchanged from 003).
- A fold without any window of a species (few recordings): that species is left out of that fold's macro-F1, and the
  table says so.
- Training is not byte-reproducible across computers: all variants of the comparison run on one computer, and the
  interval covers seeds as well as folds.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: On a window whose folder label is a species, the BirdNET target MUST be BirdNET's confidences for the
  project's species, renormalised over them, with no background share, then softened by T; when all are below 0.1,
  BirdNET's part MUST be skipped (as in 003).
- **FR-002**: On a background window the target MUST be as in feature 003.
- **FR-003**: Unit tests MUST cover FR-001 and FR-002 (the scenario of US1, the skip, the background window).
- **FR-004**: A comparison command MUST train, on 4 recording-level folds of the 006 test bundle and 3 seeds each,
  these variants: without BirdNET; with BirdNET (new target) at weight 0.5 and 0.25; with BirdNET (003 target) at
  0.5; each from scratch and from WrenNet; and MUST report per variant the test macro-F1 and the false-detection rate
  (device's threshold) with 95 % intervals over folds × seeds, and the paired difference to "without BirdNET".
- **FR-005**: The comparison's results MUST be committed (a results file and a summary), and the README, `docs/
  PIPELINE.md` and the 003 documents MUST state the conclusion they support, replacing the 003 comparison.
- **FR-006**: With the option off, training MUST be unchanged (same `train.json` and `history.csv` as before on the
  same computer).

### Key Entities

- **Variant**: init (scratch, WrenNet) × BirdNET (off, new target at 0.5 / 0.25, 003 target at 0.5).
- **Fold**: a fifth of the bundle's recordings, all windows of a recording together.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On `tests/birds_birdnet`, "target's top class = folder" is at least 90 % for every species whose windows
  BirdNET hears above 0.1 (it was 50–100 %).
- **SC-002**: The comparison covers 8 variants × 4 folds × 3 seeds = 96 trainings, each with its result recorded.
- **SC-003**: Every difference the documentation calls an improvement or a loss has a 95 % interval excluding zero;
  otherwise it is reported as "no measurable difference".
- **SC-004**: Option off: training identical to before (FR-006).

## Assumptions

- Feature 006 (the test bundle) is merged first; feature 009 (training experiments) may later run this comparison with
  its own tooling, but this feature does not depend on it.
- 96 short trainings fit in a few hours on a laptop (about 1 minute each measured for these data sizes); the
  comparison runs on demand, not in CI. CI runs the unit tests.
- Folds are fixed by a seed and recorded, so the comparison can be rerun on the same split.
