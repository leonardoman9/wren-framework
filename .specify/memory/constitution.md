# Wren Framework Constitution

## Core Principles

### I. Biologist-First Design

Every interface — YAML configuration, error messages, reports, documentation — MUST be
understandable by a field biologist, ornithologist, or ecologist with no programming experience
beyond running a terminal command. Technical jargon (tensors, quantisation, ONNX, TFLM) MUST
NOT appear in user-facing output; it belongs only in developer logs and internal documentation.
The pipeline's single entry point is `python -m wren run project.yaml`.

### II. One YAML, One Command, One Firmware

A user MUST be able to go from audio folders to a flashable AudioMoth firmware binary by writing
one `project.yaml` file and running one command. Every parameter MUST have a documented default
so that a minimal YAML (data folder, class names, negative class) is sufficient. Advanced
parameters (learning rate, augmentation, loss function, device schedule) are optional overrides,
not requirements.

### III. Fixed Architecture, Variable Classes

The neural network architecture is WrenNet (CNN–GRU with attention): fixed topology, fixed
operator set, fixed 32 kHz sample rate, fixed 3 s classification window. The number and names
of classes, the learned filterbank parameters, and the classifier weights are the only things
that change between projects. This constraint guarantees that every model the pipeline produces
fits on the AudioMoth's Cortex-M4 and has been verified against the device.

### IV. End-to-End Verification

Every stage of the pipeline — prepare, train, evaluate, export, package, firmware — MUST
produce machine-checkable artefacts (JSON reports, checksums, parity results). The pipeline
MUST NOT produce a firmware binary without verifying:
- front-end + core = whole model (logit parity);
- ONNX = PyTorch (logit parity);
- int8 TFLite = float within tolerance;
- device pipeline (C front-end + int8 backbone + C tail) = PyTorch on test windows;
- firmware flash fits in 245,760 bytes;
- firmware passes its fault-scenario and schedule tests.

A stage that fails its check MUST stop the pipeline with a clear, non-technical error message.

### V. Reproducibility

Given the same `project.yaml`, the same audio files, and the same pinned dependency versions:

- **Docker reference path**: the `linux/amd64` image built from the pinned Dockerfile MUST
  produce byte-identical outputs on any host. Its deterministic stages (export, package,
  firmware) MUST be bit-exact.
- **Native installs** (Linux x86_64, macOS arm64): the pipeline MUST complete and pass every
  parity check and quality gate (Principle IV and Pipeline & Quality Gates), but its outputs are
  NOT required to be byte-identical to the Docker reference, because float conversion and
  compiler builds differ by CPU architecture and operating system.
- **Training**: non-deterministic stages (e.g. training on GPU or across platforms) MUST
  document their tolerance and are judged by the quality gates, not by bytes.
- **Shipped bird model**: it MUST pass a byte-exact reproduction test on every CI run, executed
  inside the Docker image.

Rationale: a single pinned execution environment is the only place byte-identity can be
guaranteed; native installs keep the same correctness guarantees through the parity checks.

### VI. Honesty in Reporting

The report MUST state what was measured and what was not. It MUST NOT extrapolate accuracy
numbers to conditions not tested (different species, different sites, different recording
hardware). Known limitations — distribution shift between studio recordings and field
conditions, lack of long-recording false-positive data, untested species — MUST be named, not
hidden. If the model's quality gate is barely met, the report MUST say so.

## Target Audience

The primary users are wildlife researchers deploying passive acoustic monitoring with AudioMoth
devices. They typically:
- have audio recordings organised in folders by species;
- know which species they want to detect and what "background" sounds like at their sites;
- do not know Python, machine learning, or embedded systems;
- need a firmware file they can flash with the AudioMoth Flash App;
- need a report they can cite in a paper or attach to a funding application.

Secondary users are ML engineers or conservation-technology developers extending the framework
(adding architectures, targets, or augmentation methods). They interact with the Python API
and the firmware build system, not with `project.yaml`.

## Pipeline & Quality Gates

The pipeline stages run in order: `prepare → train → evaluate → export → package → firmware →
report`. Each stage:
- reads its inputs from the previous stage's output directory;
- writes to its own output directory under `runs/<name>/`;
- is skipped when its inputs have not changed (content hash);
- can be forced to rerun with `--force`;
- logs to `runs/<name>/logs/`.

Quality gates between stages:
- **evaluate → export**: test macro-F1 MUST exceed 1/N + 0.1 (where N is the class count).
- **package → firmware**: int8 device-pipeline macro-F1 MUST be within 0.05 of float.
- **firmware → report**: flash MUST fit; all firmware tests MUST pass.

A failed gate stops the pipeline and names the failing metric in the error message.

## Governance

This constitution supersedes ad-hoc decisions in code comments, commit messages, or
conversation logs. Amendments require:
1. a description of what changes and why;
2. a version bump following semantic versioning (MAJOR for principle removal or redefinition,
   MINOR for new principles or material expansion, PATCH for clarifications);
3. review by a project maintainer.

All pull requests MUST be checked against the principles above. A PR that violates a principle
MUST either fix the violation or propose a constitution amendment in the same PR.

**Version**: 2.0.0 | **Ratified**: 2026-10-22 | **Last Amended**: 2026-10-22
