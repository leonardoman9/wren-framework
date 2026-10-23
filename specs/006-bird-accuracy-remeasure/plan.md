# Implementation Plan: Re-measure the bird model's accuracy

**Branch**: `006-bird-accuracy-remeasure` | **Date**: 2026-10-23 | **Spec**: [spec.md](spec.md)

## Summary

A committed test bundle `examples/birds/test/` holds freely licensed xeno-canto recordings of the bird model's 8
species (the stretches outside the calibration clips, ≤ 120 s per recording) and BirdNET-screened CC0 Freesound field
recordings with no birds, all with attribution and checksums, built once by a maintainer tool
(`tools/make_test_bundle.py`, xeno-canto and Freesound API keys). One command, `python tools/measure_bird_accuracy.py`,
runs the pipeline's own pieces on it — prepare's call-finding (with a new "ratio 0 = none" split option), the float
PyTorch reference, and the device pipeline (C front-end + shipped int8 backbone + C tail) — and writes `accuracy.json`
and `accuracy.md`: accuracy, macro-F1, per-class recall/precision, confusion matrix, false-detection rate on non-bird
windows, recording-level bootstrap intervals, and the subset of recordings uploaded after WrenNet's training
(2025-06-11). The committed `accuracy.json` is the reference that a Docker CI step reproduces byte for byte. §9 of
`audiomoth/docs/VERIFICATION.md` and the model package README get the new figures.

## Technical Context

**Language/Version**: Python 3.11 (torch and tf environments, unchanged locks) · **Dependencies**: none new (BirdNET
from feature 003 for screening, maintainer only) · **Storage**: `examples/birds/test/` (~45–55 MB of FLAC, like the
calibration bundle's 47 MB) · **Testing**: `tests/test_bird_accuracy.sh` (Docker: byte-exact `accuracy.json`),
`tests/test_measure.py` (statistics on a fixture), `tests/test_examples.py` (attribution and licences of the new
bundle), `tests/test_training.py` (prepare unchanged for ratios > 0) · **Constraints**: the shipped package and every
pipeline output unchanged (constitution V); no BioDCASE; CC0/BY/BY-SA only · **Blocker**: XC_API_KEY and
FREESOUND_API_KEY for the one-time bundle build (research R8)

## Constitution Check

| Principle | Effect | Status |
|-----------|--------|--------|
| I. Biologist-first | §9 and the package README in plain words, with limits stated first | Pass |
| IV. Verification | adds a measured, reproducible accuracy for the shipped model; no gate changes | Pass |
| V. Reproducibility | measurement deterministic; byte-exact in Docker (CI); bundle by checksum | Pass |
| VI. Honesty in reporting | intervals per recording, unseen subset, overlap with training and calibration stated, species below 5 recordings marked not meaningful, old analogue rows removed with a reason | Pass |
| Pipeline & Quality Gates | prepare's split change keeps every ratio > 0 behaviour; stages and gates untouched | Pass |

Post-design re-check: pass.

## Project Structure

```text
specs/006-bird-accuracy-remeasure/{plan,research,data-model,quickstart}.md, contracts/{test-bundle,measurement}.md, tasks.md

tools/make_test_bundle.py              # new, maintainer: xeno-canto + Freesound -> examples/birds/test/
tools/measure_bird_accuracy.py         # new: the one command (stdlib + subprocess)
wren/torchside/accuracy.py             # new: statistics (bootstrap per recording) -> accuracy.json / accuracy.md
wren/torchside/prepare.py              # split ratio 0 = no carving for that split
wren/tfside/device_eval.py             # optional per-window output (labels, float and int8 probabilities)
examples/birds/test/                   # new bundle: audio/, ATTRIBUTION.csv, SHA256SUMS, LICENSE.md, README.md, accuracy.json
tests/test_bird_accuracy.sh            # new: rerun and compare accuracy.json (byte-exact in Docker)
tests/test_measure.py                  # new: statistics on a fixture (bootstrap, false detections, minimum recordings)
tests/test_examples.py                 # the new bundle's attribution and licences
.github/workflows/ci.yml               # unit test + Docker step
audiomoth/docs/VERIFICATION.md §9, audiomoth/firmware/bird/model/README.md, README.md ("Measured so far"), docs/PIPELINE.md
```

## Complexity Tracking

No constitution violations.
