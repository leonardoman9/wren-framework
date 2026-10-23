# Feature Specification: Re-measure the bird model's accuracy

**Feature Branch**: `006-bird-accuracy-remeasure`

**Created**: 2026-10-23

**Status**: Draft

**Input**: User description: "6 — re-measure the shipped bird model's accuracy: the figures in
audiomoth/docs/VERIFICATION.md §9 were measured on the previous package (different calibration audio). Measure the
current package (float and device int8) on freely licensed labelled audio (xeno-canto CC0/BY/BY-SA, never BioDCASE) and
replace the old figures, reproducibly."

## Context

`audiomoth/docs/VERIFICATION.md` §9 reports 76.4 % (float) and 75.8 % (device) on 161 windows of the 8 species, from
files of the project Drive, with scripts that live in the WrenNet repository. Those figures:

- were measured on the previous package (feature 001 recalibrated the int8 model on `examples/birds/calibration/`);
- used files that were "probably also training files", picked windows by loudness, and had no non-bird windows;
- cannot be reproduced from this repository.

Freely licensed audio is scarce (measured 2026-10-22): of 12,552 xeno-canto recordings of the 8 species, 82 are CC0,
CC BY or CC BY-SA, and 76 of them are already in the calibration bundle (4–16 per species; only 4 each for *Emberiza
cia* and *Poecile montanus*).

## Clarifications

### Session 2026-10-23

- Q: Where do the figures go? → A: They replace §9 of `audiomoth/docs/VERIFICATION.md` and the model package README's
  accuracy line; the old figures stay in git history, and the new section says what changed in the method.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Honest, current accuracy figures for the shipped bird model (Priority: P1)

A researcher deciding whether to deploy the bird model reads its accuracy in the documentation. The figures describe
the package that ships today, say exactly what audio they were measured on, and give their uncertainty and limits.

**Why this priority**: Constitution VI: the report must state what was measured. Today's figures describe a package
that no longer ships.

**Independent Test**: Run the measurement; the documented numbers equal the measurement's output.

**Acceptance Scenarios**:

1. **Given** the current package, **When** the measurement runs, **Then** it reports accuracy and macro-F1 for the
   float model and the device version (C front-end + int8 backbone + C tail), overall and per species, with the
   number of windows and recordings behind each figure.
2. **Given** the documentation, **When** a reader looks up the accuracy, **Then** they find these figures, the audio's
   source and licences, the uncertainty, and the known limits (sample size, possible overlap with training data).

---

### User Story 2 — Anyone can repeat the measurement (Priority: P1)

A reviewer clones the repository and runs one command. It produces the same figures: the test audio is in the
repository (or fetched by a checksum-verified tool), and in the Docker image the result is identical.

**Why this priority**: Figures that cannot be reproduced cannot be cited (constitution V, VI).

**Independent Test**: Run the command twice and in the Docker image; the outputs match.

**Acceptance Scenarios**:

1. **Given** a fresh clone with the setup done, **When** the reviewer runs the measurement command, **Then** it
   completes and prints the same figures as the documentation.
2. **Given** the Docker image, **When** the measurement runs, **Then** its output is byte-identical between runs and
   hosts.

### Edge Cases

- A species with very few test recordings: its figure is reported with its count and interval, and flagged as not
  meaningful below a stated minimum, instead of being hidden or averaged away.
- A recording whose licence changes or is withdrawn on xeno-canto: the committed attribution and checksums keep the
  measurement fixed; a later refresh is a new measurement.
- Windows from one recording are correlated: the uncertainty is computed per recording, not per window.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The test audio MUST be freely licensed (CC0, CC BY, CC BY-SA), with one attribution row per clip, and
  MUST contain no BioDCASE audio.
- **FR-002**: The test audio MUST come from [NEEDS CLARIFICATION: only 6 free recordings of the 8 species are not
  already in the calibration bundle. Which audio is the test set?]
- **FR-003**: The test set MUST [NEEDS CLARIFICATION: include non-bird windows (so the "non-bird" class and false
  detections are measured), or only the 8 species as before?]
- **FR-004**: The measurement MUST report, for the float model and the device version: accuracy, macro-F1, per-species
  recall and precision, the confusion matrix, and the number of windows and recordings, with intervals computed per
  recording.
- **FR-005**: Windows MUST be chosen by the pipeline's own call-finding (as in training projects), not by loudness, and
  the choice MUST be deterministic.
- **FR-006**: The measurement MUST run with one command from the repository and give identical output on repeated runs;
  in the Docker image identical on any host.
- **FR-007**: §9 of `audiomoth/docs/VERIFICATION.md` and the model package README MUST show the new figures, the
  method, the audio's source and the limits, and MUST no longer present the old figures as current.
- **FR-008**: The documentation MUST state that WrenNet's training data came from xeno-canto and that overlap with the
  test recordings cannot be excluded unless the test recordings are newer than the training data, and which applies.

### Key Entities

- **Test bundle**: the clips, their attribution (recordist, licence, source, start time), checksums and labels.
- **Accuracy report**: per model version (float, device), overall and per species figures with counts and intervals.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: One command reproduces every figure in §9; two runs give identical output.
- **SC-002**: Every species has at least the stated minimum of test recordings, or its figure is marked as not
  meaningful.
- **SC-003**: The difference between the float model and the device version is reported with its interval, so the
  cost of the device conversion is known on current data.
- **SC-004**: No figure in the documentation is older than the package it describes.

## Assumptions

- The measurement uses the shipped package as it is; it does not retrain or recalibrate anything.
- xeno-canto's API (with the maintainer's key) is used only by a maintainer tool that builds the test bundle once, as
  for the calibration bundle; users never need a key.
- The simulated analogue-chain rows of the old §9 (70/60/50 dB SPL) are re-measured too if the simulation can be
  reproduced from this repository; otherwise they are removed and the section says so.
