# Feature Specification: Training experiments, then deploy the best

**Feature Branch**: `009-training-experiments`

**Created**: 2026-10-23

**Status**: Draft

**Input**: User description: "Training experiments separate from deployment: run many trainings of a project (different
settings and seeds) without converting or building firmware, keep each one with its settings and validation results,
compare them, pick the best by validation (never by test), and then promote that one to the rest of the pipeline
(export, package, firmware, report), where the test set is used once. Optionally an automatic hyperparameter search
(e.g. Optuna) over the training settings, with early stopping of bad trials."

## Context

Today `python -m wren run project.yaml` trains once and goes on to the firmware; `--stages prepare,train,evaluate`
stops before the conversion, but every new training overwrites the previous one, nothing keeps the tries side by side,
and `evaluate` reports the test set each time, which invites choosing on the test set. With few recordings per species
the result of one training also depends a lot on the seed (feature 003 measured 0.59 and 0.79 for the same project on
two computers). Researchers need to try settings, compare them fairly, and deploy the best.

## Clarifications

### Session 2026-10-23

- Q: What is "best"? → A: The highest validation macro-F1, averaged over the seeds of the same settings (ties: lower
  validation loss). The test set is never shown while experimenting; it is measured once, on the promoted trial.
- Q: Few recordings: one validation split is noisy. → A: Experiments can use cross-validation by recording group
  (k folds, each recording in one fold); the default is the project's single split, with 3 seeds per setting.
- Q: Automatic search? → A: Yes, optional, with a budget (number of trials or time), stopping trials early whose
  validation curve is clearly worse than the best so far; it searches only the training settings, never the
  architecture (constitution III). The tool is chosen in planning (Optuna is the candidate); it is installed only for
  this option.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Try several training settings and seeds without building firmware (Priority: P1)

A researcher lists, in `project.yaml`, the settings they want to try (for example two learning rates, with and without
learning from BirdNET) and how many seeds. One command trains every combination, keeps each trial with its settings,
seed, validation curve and result, and prints a comparison table. Nothing is converted or built.

**Why this priority**: The core of the request: experimenting without paying for conversion and firmware each time,
and without losing earlier tries.

**Independent Test**: On the synthetic project, an experiment of 2 settings × 3 seeds produces 6 kept trials and one
table ranked by mean validation macro-F1 with its spread.

**Acceptance Scenarios**:

1. **Given** an experiment of N settings × S seeds, **When** it runs, **Then** N × S trials are kept, each with its
   settings, seed, curve, best epoch and validation scores.
2. **Given** an experiment already partly run, **When** it runs again, **Then** finished trials are reused and only the
   missing ones train.
3. **Given** the comparison table, **When** a researcher reads it, **Then** it shows validation figures only, mean and
   spread per setting, and no test figure.

---

### User Story 2 — Promote the best trial and deploy it (Priority: P1)

The researcher promotes the best setting (or a trial they choose). The pipeline then takes that trial's checkpoint
through evaluate (the test set, used once), export, package, firmware and report, exactly as a normal run would.

**Why this priority**: Experiments are worthless unless the chosen model reaches the device the same way as today.

**Independent Test**: After promotion, `python -m wren run` builds the firmware from the promoted checkpoint; the
report says which trial it is, how many were tried, and that the choice was made on validation.

**Acceptance Scenarios**:

1. **Given** a finished experiment, **When** the researcher promotes "best", **Then** the trial with the highest mean
   validation macro-F1 (its best seed) becomes the project's model.
2. **Given** a promoted trial, **When** the pipeline runs, **Then** every check and quality gate applies as for a normal
   training, and the test set is measured once.
3. **Given** the report, **When** it describes training, **Then** it names the promoted trial and its settings, the
   number of trials and settings tried, and states that selection used validation only (constitution VI).

---

### User Story 3 — Let the computer search the settings (Priority: P2)

Instead of listing combinations, the researcher sets a budget (e.g. 40 trials or 2 hours) and which settings may vary
(or accepts the documented default ranges). The search tries settings, stops clearly bad trials early, and ends with
the same comparison table and promotion as US1–US2.

**Why this priority**: Saves the researcher's time on larger projects; the manual way (US1) already gives the benefit.

**Independent Test**: On the synthetic project, a search of 10 trials finishes, stops at least one trial early, and
promotes a trial whose validation score is the best of those run.

**Acceptance Scenarios**:

1. **Given** a budget, **When** the search runs, **Then** it never exceeds it and records every trial, stopped or not.
2. **Given** an interrupted search, **When** it is started again, **Then** it continues from the recorded trials.

### Edge Cases

- A setting outside the allowed values: refused in the project check, before any training, in plain words.
- A trial fails (e.g. out of memory): recorded as failed with its log; the experiment continues.
- Trials run on different computers: the table records the computer of each and warns when they are mixed, since
  training is not byte-reproducible across computers.
- The prepared windows change (new recordings): earlier trials are marked stale and not mixed into the comparison.
- Disk: each trial keeps its checkpoint (about 0.3 MB) and curve; no audio is copied.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `project.yaml` MUST be able to describe an experiment: lists of values for training settings, the number
  of seeds, and optionally k-fold cross-validation by recording group; every setting it may vary is a documented
  training option.
- **FR-002**: One command MUST run the experiment (prepare once, then the trials), keep every trial (settings, seed,
  fold, curve, best epoch, validation scores, computer, duration, log) and reuse finished trials on a re-run.
- **FR-003**: The comparison MUST rank settings by mean validation macro-F1 over seeds (and folds), show the spread,
  and show no test figure.
- **FR-004**: Promotion MUST make one trial's checkpoint the project's model, after which the existing pipeline
  (evaluate, export, package, firmware, report) runs unchanged on it, with every check and gate.
- **FR-005**: The test set MUST be evaluated only on the promoted model; the experiment never computes or shows test
  results.
- **FR-006**: The report MUST name the promoted trial and its settings, the number of trials and settings tried, and
  that the selection used validation only.
- **FR-007**: An optional automatic search MUST take a budget (trials or time), the settings it may vary (with
  documented default ranges), stop trials whose validation curve is clearly worse than the best so far, never vary the
  architecture, and record every trial like FR-002.
- **FR-008**: Without an experiment in `project.yaml`, `python -m wren run` MUST behave exactly as today
  (constitution II).
- **FR-009**: User-facing words MUST be plain ("try several training settings", "keep the best"); terms like
  hyperparameter, pruning or TPE only in the technical documentation (constitution I).

### Key Entities

- **Experiment**: the settings to vary, seeds, folds or search budget; belongs to a project.
- **Trial**: one training with fixed settings, seed and fold; its checkpoint, curve, scores, computer, status.
- **Promotion**: the choice of one trial as the project's model, recorded with the reason.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A 2 × 3 experiment on the synthetic project runs with one command and keeps 6 trials; a re-run trains 0.
- **SC-002**: After promotion, the firmware is built from the promoted checkpoint and passes every check.
- **SC-003**: No test figure appears anywhere before promotion (checked by a test on the experiment's outputs).
- **SC-004**: A search with a budget of 10 trials on the synthetic project stops at least one trial early and ends
  within its budget.
- **SC-005**: Projects without an experiment train exactly as before (same `train.json` and `history.csv`).

## Assumptions

- Feature 005 (dependency clean-up) is done first, so that adding the search tool to the torch environment is one
  lock change on a clean base.
- Trials run one after another on one computer by default; running them in parallel is out of scope.
- Feature 008's comparison can be expressed as an experiment of this feature later; this feature does not depend on
  it.
