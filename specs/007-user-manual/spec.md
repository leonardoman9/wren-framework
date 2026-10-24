# Feature Specification: The wren-framework manual

**Feature Branch**: `007-user-manual`

**Created**: 2026-10-23

**Status**: Draft

**Input**: User description: "As the last task of the project: a manual for wren-framework with the technical
explanation, how it works, how to use it, diagrams and architectures (pipeline, model, firmware, verification), and
whatever else a reader needs — for biologists using it and for developers extending it."

## Context

Today the knowledge is spread over a 250-line README, `docs/PIPELINE.md`, the firmware's README and three firmware
documents (`VERIFICATION.md`, `FIRST_BOARD_TEST.md`, `TFLM_BUILD.md`), the specification folders of seven features and
the code's docstrings. Each is accurate, none tells the whole story in order, and there is no picture of how the parts
fit. This feature is the project's last one: it describes the state after features 001–006 (and 005), so it comes
after them.

## Clarifications

### Session 2026-10-23

- Q: Where does the manual live and in which form? → A: In the repository, as a set of linked chapters that read well
  on GitHub, with diagrams that GitHub draws from text (so they change with the code, in the same pull request), plus
  one generated printable file (PDF) of the whole manual for offline reading and citation. The README stays the short
  entry point and links to it.
- Q (maintainer, 2026-10-23): What must the technical part explain in depth? → A: The preprocessing, the hardware
  device, the training, the model architecture, the embedded part, and the whole path from the computer to the
  firmware; plus every parameter and configurable thing, and what is fixed and why. Added as FR-003a–FR-003g and FR-010.
- Q (maintainer, 2026-10-23): Which further topics? → A: collecting training data, from device logs to ecological
  data, when to use wren or an alternative, the energy budget, and the code map with how to contribute (FR-011–FR-015).
  Not added: deployment planning, validating a deployment, end-to-end tutorials, privacy and legal, history.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — A field biologist goes from recordings to a deployed AudioMoth (Priority: P1)

An ornithologist who has never used the project reads the manual's user part from the start. It tells them what the
tool can and cannot do, how to install it, how to organise their recordings, how to write `project.yaml`, how to read
the report, how to flash and configure the AudioMoth, how to read its log and LEDs, and what to do when something
fails. They reach a working device without reading code or asking for help.

**Why this priority**: These are the people the constitution names first (principle I); the manual exists for them.

**Independent Test**: A person outside the project follows the user part on the synthetic example and on the bird
example; they reach a firmware file and can explain the report's main figures, using only the manual.

**Acceptance Scenarios**:

1. **Given** the user part, **When** a biologist follows it on `tests/synthetic` data, **Then** every command it shows
   works as printed and every output it describes appears.
2. **Given** a run that stops with one of the pipeline's messages, **When** the biologist looks it up in the manual,
   **Then** they find what it means and what to do.
3. **Given** the manual's "is this tool right for me" section, **When** a reader has a use case outside its limits
   (thousands of species, sounds longer than 3 s, no AudioMoth), **Then** the manual says so plainly.

---

### User Story 2 — A developer understands every part in depth, well enough to change it (Priority: P1)

A conservation-technology developer wants to add a class of sounds, change the training, or port the firmware. The
manual's technical part explains, with diagrams, the pipeline's stages and their keys, the two environments, the
WrenNet model and its split into front-end, backbone and tail, the int8 conversion, the firmware's tasks, memory and
schedule, the verification chain (each parity check and what it proves), what is byte-reproducible and why, and how
to change each part safely. The preprocessing, the hardware, the training, the architecture, the embedded part and the
path from the computer to the firmware are each explained in depth (FR-003a–FR-003g), and every parameter is listed with
its default and effect (FR-010).

**Why this priority**: Without it every change starts by reverse-engineering; it is the second audience the
constitution names.

**Independent Test**: For each of a set of change scenarios (add a training option, change the model code, change a
dependency, change the firmware schedule), the manual names the files involved, the checks that must keep passing and
the command that proves it.

**Acceptance Scenarios**:

1. **Given** the technical part, **When** a developer looks for where a stage's output comes from, **Then** a diagram
   and a table lead them to the module and its inputs.
2. **Given** a proposed change to the model code, **When** the developer reads "changing things safely", **Then** they
   learn which stages re-run, which reproduction test guards it and how to update a reference on purpose.

---

### User Story 3 — A reviewer or funder checks what is claimed and how it is measured (Priority: P2)

A reviewer of a paper, or a funding body, reads the manual's evidence part: what the shipped bird model achieves on a
freely licensed test set, with intervals, the device-vs-original comparison, the firmware's measured flash, RAM, timing
and power, the known limits and open points, the licences of every component and data set, and how to cite.

**Why this priority**: Constitution VI; useful, but builds on the other two parts.

**Independent Test**: Every figure in the evidence part names the command or document that reproduces it.

**Acceptance Scenarios**:

1. **Given** the evidence part, **When** a reviewer looks for the accuracy of the bird model, **Then** they find the
   figures of feature 006 with their intervals, test set and limits, and the command that reproduces them.
2. **Given** the licences chapter, **When** a reader checks a component (code, firmware, WrenNet, BirdNET, calibration
   and test audio), **Then** its licence and any condition (attribution, non-commercial) are stated.

### Edge Cases

- A figure or command changes after the manual is written: the manual quotes commands and outputs that a check runs,
  so a mismatch fails CI instead of going stale silently (FR-007).
- A diagram drifts from the code: diagrams are text in the repository, reviewed in the same pull request as the code.
- A reader without internet or GitHub: the printable file contains the whole manual, diagrams included.
- Duplicated text between README, `docs/PIPELINE.md` and the manual: each fact has one home; the others link to it.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The manual MUST have three parts in this order: using wren (biologists), how wren works (developers), and
  evidence and limits (reviewers), each readable on its own, with a short "start here" page that sends each reader to
  their part.
- **FR-002**: The user part MUST cover: what wren does and does not do; installation (native and Docker) and
  troubleshooting; preparing recordings; every `project.yaml` option with its default and when to change it (including
  learning from BirdNET); running and re-running; reading the report; the device (flashing, `CONFIG.TXT`, schedule,
  log, LEDs, SD card, battery life); the first board test; a glossary of every term it uses; and every message the
  pipeline can stop with, with its meaning and remedy.
- **FR-003**: The technical part MUST cover with diagrams: the overall architecture; the stage graph with each stage's
  inputs, outputs and key; the two Python environments and why; the WrenNet model (front-end, backbone, GRU, attention,
  classifier) and the device split; the int8 conversion and its checks; the firmware (tasks, memory map, external SRAM,
  schedule, file system, failure handling); the verification chain from PyTorch to the board; what is pinned and why
  the bytes match; the tests and CI; and how to change each part safely.
- **FR-003a — Preprocessing**: how audio becomes training windows, step by step: decoding and supported formats,
  downmix, resampling to 32 kHz, the device's high-pass filter, short files, long files, how call windows are found
  (per-class frequency bands learned from the training files, event prominence over the file's noise floor, spacing,
  per-file limits), background windows, splitting by file or by recording group, duplicate detection, every warning and
  error prepare can give and what it means, with a worked example on one recording.
- **FR-003b — Hardware**: the AudioMoth Dev as the model sees it: microcontroller (EFM32WG380F256, Cortex-M4F at 48 MHz,
  256 KB flash, 32 KB RAM), external SRAM and its 8-bit bus, microphone, analogue gain and ADC, sample rate, SD card,
  real-time clock, LEDs, battery and power states, with a block diagram and the limits each one sets (model size,
  window length, speed, recording time, battery life).
- **FR-003c — Training**: what WrenNet learns and how: starting points (`init: wrennet` with the frozen learned
  filterbank, or from scratch), batch composition and the background share, every augmentation, the loss (cross-entropy,
  focal, and the focal distillation from BirdNET with its corrected target), learning-rate schedule, warm-up, model
  selection and early stopping, the quality gates, what the training report fields mean, and how to tell over-fitting
  or too little data.
- **FR-003d — Architecture**: the WrenNet model layer by layer with shapes: the learned log-linear spectrogram front-end
  (STFT, 64 filters, breakpoint and transition width), the PhiNet/MatchboxNet convolutional backbone, the GRU, the
  attention pooling and the classifier; parameter counts and operation counts; why this architecture is fixed
  (constitution III) and what changes between projects (classes, filterbank, weights).
- **FR-003e — Embedded part**: how the model runs on the device: the split into C front-end, int8 TFLite-Micro backbone
  and C tail (GRU, attention, classifier) and why; TFLite-Micro, the registered operators and the local patches; the
  memory plan (flash, RAM, external SRAM, arena); the firmware application (recording window, classification schedule,
  file system and logs, LEDs, watchdog, failure handling and recovery); the diagnostic build; and the simulator, QEMU
  and host tests that check it.
- **FR-003f — From computer to firmware**: the whole path with one diagram and one section per step: checkpoint →
  front-end/core split → ONNX → simplification → TensorFlow → int8 TFLite calibrated on the project's audio → the C
  tables of the front-end and the C weights of the tail → the model package (headers, class list, test vector) → the
  firmware build with the ARM compiler → the firmware tests → the `.bin` and the report; at each step what goes in,
  what comes out, which check proves it is right, and which file to look at when it fails.
- **FR-003g — Reproducibility**: what is pinned (environments, compiler, TFLM commit and patches, Docker image, Debian
  snapshot) and why the bytes match on every x86 computer in Docker but not between native installs.
- **FR-004**: The evidence part MUST cover: the bird model's accuracy (feature 006) with intervals and limits; the
  device-vs-original agreement; measured flash, RAM, timing and power; robustness tests (faults, long runs); the
  BirdNET comparison as measured; known limits and open points; the licences of every component and data set; how to
  cite the project, WrenNet and BirdNET.
- **FR-005**: Diagrams MUST be text in the repository that GitHub draws (no image that cannot be edited), and MUST
  appear in the printable file.
- **FR-006**: A printable file of the whole manual MUST be produced by one command from the repository and attached to
  each release; it is not committed.
- **FR-007**: Every command shown as runnable and every quoted figure MUST be checked automatically: commands by a test
  that runs them (or a documented reason why not, e.g. needs a board), figures by comparison with the file that
  produces them (e.g. `accuracy.json`), so the manual cannot drift unnoticed.
- **FR-008**: Each fact MUST have one home: the README becomes the short entry point linking to the manual, and
  `docs/PIPELINE.md` and the firmware documents are either moved into the manual or linked from it, with no
  contradicting copies.
- **FR-010 — Parameters reference**: one chapter MUST list every configurable thing, grouped by where it is set, each
  with its default, allowed values, effect and when to change it: every `project.yaml` key (data, training including
  BirdNET, calibration, test, device, firmware, envs, runs); the device's `CONFIG.TXT`; the setup and run command-line
  options; environment variables (e.g. `WREN_MIN_DISK_GB`, `WREN_PATH`); firmware build options (e.g. the diagnostic
  build); and, separately, the values fixed by the architecture or the hardware (32 kHz, 3 s windows, 64 filters, flash
  and RAM limits, at most 180 classes, class-name rules) with the reason each is fixed. The list MUST be checked
  against the code so a new option cannot be added without its entry (FR-007).
- **FR-011 — Collecting training data** (user part): how many recordings per class and why (with the quality gate's
  arithmetic), recording the background at one's own sites, keeping one recording (or one site, one day) out of both
  training and test (`data.group`), unbalanced classes and the background share, archive recordings versus field audio
  (distribution shift), labelling mistakes and how prepare's warnings reveal them.
- **FR-012 — From device logs to ecological data** (user part): the formats of `LOG.TXT`, `SESSIONS.TXT` and
  `STATE.TXT` field by field; choosing the confidence threshold (the report's suggestion, the trade-off between missed
  calls and false detections); turning per-window detections into presence/absence and activity per hour and per site;
  estimating and correcting false detections (the false-detection rates of feature 006, checking a sample by ear with
  the diagnostic build); with a small worked example from a log to a table.
- **FR-013 — When to use wren, or something else** (evidence part): an honest comparison with BirdNET and BirdNET-Pi,
  Acoupi, Edge Impulse, commercial on-device recorders such as Sonalyze's Sona One (STM32N6 with a neural accelerator,
  cellular uplink, swappable models, rental and cloud platform; poster at the BioDCASE Workshop, 23 October 2026), and
  continuous recording followed by analysis on a computer: species coverage, accuracy, cost,
  power, need for a computer in the field, privacy, effort; when each is the better choice.
- **FR-014 — Energy budget** (technical and evidence parts): energy per classification, per recording window and in
  sleep; battery life for common batteries and schedules as a worked calculation; which figures are measured on a board
  and which are estimates (constitution VI: an unmeasured figure is labelled as such).
- **FR-015 — Code map and contributing** (technical part): which file does what (pipeline, torch side, tf side,
  firmware, tools, tests), how a feature is made in this repository (spec-kit: `specs/`, the constitution, spec → plan
  → tasks → implementation → pull request), the checks a pull request must pass, and how to update a byte-exact
  reference (bird package, firmware hashes, accuracy) when a change is intended; and the constitution's rule (v2.1.0) that from
  this manual's first version on, every pull request updates the manual when it changes anything the manual describes.
- **FR-009**: The user part MUST follow constitution I (no machine-learning or embedded jargon without an explanation
  in the glossary); the technical part may use technical terms.

### Key Entities

- **Manual**: the three parts, the start page, the glossary and the printable file.
- **Diagram**: a text-defined figure stored with the chapter that uses it.
- **Checked example**: a command or figure in the manual with the test that keeps it true.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A biologist new to the project produces a firmware file from the synthetic example in under one hour
  using only the user part (timed once with a person outside the project, or by following it literally on a clean
  machine).
- **SC-002**: 100 % of the pipeline's stop messages are listed with a meaning and a remedy.
- **SC-003**: At least 12 diagrams: architecture, stage graph, environments, preprocessing (from a recording to
  windows), training loop, model layers with shapes, device split, computer-to-firmware path, hardware blocks, firmware
  tasks and schedule, memory map, verification chain (plus CI).
- **SC-008**: The log-to-data example of FR-012 runs from a committed sample log to its table with the command shown.
- **SC-007**: 100 % of `project.yaml` keys, command-line options, environment variables and firmware build options in
  the code appear in the parameters chapter, checked automatically; every fixed value is listed with its reason.
- **SC-004**: 100 % of runnable commands and quoted figures are covered by an automatic check or carry a stated reason
  why they cannot be.
- **SC-005**: Nothing in the README, `docs/` or the firmware documents contradicts the manual (checked in review; facts
  live in one place).
- **SC-006**: The printable file is generated by one command and contains every chapter and diagram.

## Assumptions

- Features 005 and 006 are done before this one, so the manual describes their result (dependencies, accuracy).
- The manual is in English like the rest of the repository; translations are out of scope.
- Measured hardware figures (power, timing on the board) are quoted from `audiomoth/docs/VERIFICATION.md` as they stand;
  where they are still open (§14), the manual says so rather than estimating.
- The printable file is built with tools available in the Docker image or the torch environment; no new paid service.
