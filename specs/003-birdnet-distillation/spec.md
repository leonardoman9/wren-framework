# Feature Specification: Learning from BirdNET for bird projects

**Feature Branch**: `003-birdnet-distillation`

**Created**: 2026-10-23

**Status**: Draft

**Input**: User description: "per gli uccelli si può usare birdnet. in effetti ha senso. bisogna specificare che si può
usare distillation SE si usano gli uccelli, e i dati vengono distillati da birdnet." — for birds, BirdNET can be used;
distillation must be an option only when the project's sounds are birds, and the training targets come from BirdNET.

## Context

The shipped bird model was made this way: WrenNet (small enough for the AudioMoth) learned not only from the folder
labels but also from BirdNET's opinion of every recording (knowledge distillation, BirdNET as the teacher). That code
lived in WrenNet's own repository and was removed from `wrennet/` by feature 002 (it could not run here). The wren
pipeline today trains from folder labels only. This feature brings the BirdNET teacher into the pipeline, for bird
projects only, because BirdNET only knows birds (and a few other sound classes).

## Clarifications

### Session 2026-10-23

- Q: On or off by default? → A: Off. The researcher turns it on in `project.yaml`; a project that does not ask for it
  trains exactly as today (constitution II: advanced options are optional overrides).
- Q: How does the pipeline know the sounds are birds? → A: Every sound except the background one must be a species
  BirdNET knows, identified by its scientific name (the folder name, or a name given in `project.yaml`). If any sound
  is not recognised, the pipeline stops before training and names the sounds it could not match.
- Q: Which training loss? → A: Focal distillation, as in the original WrenNet project (`FocalDistillationLoss`): a
  focal loss on the folder labels combined with a distillation loss on BirdNET's opinion,
  total = (1 − α) · focal(folder label) + α · T² · KL(model ‖ BirdNET), both softened by the temperature T. α (BirdNET's
  share), T and the focal γ have documented defaults taken from the original project and can be overridden (values:
  see below).
- Q: Which audio does BirdNET analyse? → A: Only the windows of the training set (the labelled folders), as in the
  original project. BirdNET does not label any other audio.
- Q: Is the distillation target computed exactly as in the original code, or corrected so that BirdNET's opinion
  counts? → A: Corrected (the maintainer: "correct it if the original loss was flawed"). The loss formula of the original
  is sound and stays; its teacher target was not: it passed BirdNET's confidences (independent values between 0 and 1,
  not a distribution) through softmax(confidence / T), which gives an almost uniform target (a 90 % BirdNET detection
  becomes ~13 % among 9 classes), and set "non-bird" by fixed rules (0.1, 0.3, 0.1 × other species). The new target is
  a probability distribution over the project's classes: BirdNET's confidence for each project species, plus "non-bird"
  as the confidence that none of them is present, normalised to sum 1, then softened by T (p^(1/T), renormalised).
- Q: Which defaults for α (BirdNET's share), T (temperature) and γ (focal)? → A: α 0.5, T 4, γ 2, the defaults of the
  original `FocalDistillationLoss`; γ 2 is also wren's existing `focal_gamma` default.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Train a bird model that also learns from BirdNET (Priority: P1)

An ornithologist has folders of recordings, one per species, plus a background folder recorded at their sites. They
turn on "learn from BirdNET" in `project.yaml` and run the usual single command. Before training, the pipeline asks
BirdNET what it hears in each 3 s window the model will learn from; the model then learns both the folder label and
BirdNET's graded opinion. The rest of the pipeline (test, conversion, firmware, report) is unchanged.

**Why this priority**: This is the feature. A teacher's graded opinion (e.g. "70 % Periparus ater, 20 % Poecile
montanus") carries more information than a folder label, which helps a small model when the recordings are few or
noisy.

**Independent Test**: Run a bird project with the option on and off on the same recordings; both complete every
stage; the report of the first says the model learned from BirdNET and how much, the second does not.

**Acceptance Scenarios**:

1. **Given** a bird project with the option on, **When** the user runs the pipeline, **Then** every stage completes and
   the firmware passes every check, as with the option off.
2. **Given** the same project, **When** the pipeline runs a second time with nothing changed, **Then** BirdNET's
   results are reused, not recomputed.
3. **Given** the option on, **When** the report is written, **Then** it states that the model learned from BirdNET,
   which BirdNET version, its licence, and how far BirdNET agreed with the folder labels on the training windows.

---

### User Story 2 — A clear refusal when the sounds are not birds BirdNET knows (Priority: P1)

A herpetologist with frog folders turns the option on. The pipeline stops in seconds, before any training, and says in
plain words that learning from BirdNET only works for bird species BirdNET knows, and which of their sounds it could not
match.

**Why this priority**: The option's condition ("only if birds") must be enforced, not just documented; a silent
fallback would mislead the user about how their model was made (constitution VI).

**Independent Test**: The synthetic test project (tonal sounds, not birds) with the option on stops at the project
check with that message.

**Acceptance Scenarios**:

1. **Given** a project whose sounds are not BirdNET species, **When** the option is on, **Then** the pipeline stops
   before training and lists the unmatched sounds.
2. **Given** a bird project where one folder name is a common or misspelled name, **When** the option is on, **Then**
   the message names that folder and shows how to give its scientific name.
3. **Given** any project with the option off, **When** it runs, **Then** BirdNET is never needed nor mentioned.

---

### Edge Cases

- BirdNET hears nothing in a species window (a faint or distant call): the window keeps its folder label; BirdNET's
  "nothing" does not override it.
- BirdNET hears a different species than the folder says (a second bird singing): the model learns both, weighted; the
  report counts how often this happened per species.
- Background windows: BirdNET's opinion that a target species is present in a background window is reported (it may
  reveal a mislabelled background file).
- BirdNET is not installed or cannot be downloaded: the pipeline stops with a plain message and how to install it,
  before any long step.
- Training on another computer: BirdNET's results are stored with the run, so the training stage can be repeated
  without BirdNET.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `project.yaml` MUST offer an option to learn from BirdNET in training projects, off by default; with the
  option off, training, its outputs and the report MUST be as today.
- **FR-002**: With the option on, the project check MUST verify, before any training, that every sound other than the
  background is a species BirdNET knows, matched by scientific name (folder name, or an optional per-sound name in
  `project.yaml`), and MUST stop with a plain message listing every unmatched sound otherwise.
- **FR-003**: With the option on, BirdNET MUST analyse every 3 s window of the training set (not the validation or test
  windows, which are judged by folder labels only), and
  its results MUST be stored in the run folder and reused while the windows and BirdNET's version do not change.
- **FR-004**: Training MUST use the focal distillation loss of the original WrenNet project: a focal loss on each
  window's folder label plus a distillation (KL divergence) loss towards BirdNET's opinion for the project's species and
  for "no species of interest", combined as (1 − α) · focal + α · T² · KL with temperature T. α, T and the focal γ MUST
  default to 0.5, 4 and 2 (the original `FocalDistillationLoss` defaults) and be overridable in `project.yaml`.
- **FR-004a**: BirdNET's opinion of a window MUST become a probability distribution over the project's classes before
  it is used as the target: per-species confidences plus a "non-bird" share for "none of the project's species",
  normalised to sum 1 and softened by T; the original project's softmax over raw confidences and its fixed "non-bird"
  values MUST NOT be used. A unit test MUST show that a 90 % BirdNET detection gives that species at least three times
  the uniform share (1/N) of the target at the default T, where the original target gave 1.2–1.3 times.
- **FR-005**: Model selection (best epoch) and the quality gates MUST keep using the folder labels only, so a model
  that copies BirdNET's mistakes is not rewarded (constitution, Pipeline & Quality Gates).
- **FR-006**: The report MUST state that the model learned from BirdNET, BirdNET's version and licence, α, T and γ, and per species how often BirdNET's top species agreed with the folder label on the training windows.
- **FR-007**: User-facing text (YAML option, messages, report) MUST describe the option in plain words ("learn from
  BirdNET"), without the term "distillation" outside the technical details (constitution I).
- **FR-008**: BirdNET MUST NOT be needed to convert a model, to run a project with the option off, or to build the
  firmware; only training with the option on needs it.
- **FR-009**: Test coverage MUST include the refusal path (non-bird sounds) and a bird project with the option on that
  runs to the firmware, using only freely licensed audio (no BioDCASE).

### Key Entities

- **BirdNET opinion of a window**: for one 3 s training window, BirdNET's confidence for each of the project's species
  and for "none of them"; stored per run, tied to the window and to BirdNET's version.
- **Species match**: the mapping from each project sound to a BirdNET species (scientific name), checked before
  training.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A non-bird project with the option on stops in under 30 seconds with a message that names every
  unmatched sound.
- **SC-002**: A bird project with the option on completes every stage, and its firmware passes every check.
- **SC-003**: On a bird project with few recordings per species (10 or fewer), the model that learned from BirdNET
  scores at least as well on the test windows (macro-F1) as the same project trained without it; the comparison is
  reported, not assumed.
- **SC-004**: A second run with unchanged recordings does not ask BirdNET again.
- **SC-005**: Projects with the option off produce the same training results as before this feature (same settings,
  same computer).

## Assumptions

- The researcher's species folders hold birds BirdNET covers (about 6,500 species worldwide); the background folder
  is not a species.
- BirdNET runs on the researcher's computer with no internet access needed after installation, and is installed as an
  extra only when the option is used (the base install stays as in feature 001).
- BirdNET's licence permits this use for research and education. Checked in planning: the official BirdNET v2.4
  model record (Zenodo 15050749) is licensed CC BY-NC 4.0 (the `birdnet` package README says CC BY-NC-SA 4.0); the
  report and README state the licence, so a researcher knows the non-commercial condition that may apply to a model
  trained with it.
- BirdNET listens at 48 kHz and the pipeline prepares audio at 32 kHz. Everything BirdNET uses (up to 15 kHz) is in the
  prepared 32 kHz audio, so BirdNET analyses the prepared windows resampled to 48 kHz (checked in planning: it names
  the right species on the bird calibration clips this way).
- Accuracy gains are measured, not promised (constitution VI): SC-003 is checked on a freely licensed test project and
  reported as measured.
