# Implementation Plan: Fix the BirdNET target and measure whether it helps

**Branch**: `008-birdnet-target-fix` | **Date**: 2026-10-24 | **Spec**: [spec.md](spec.md)

## Summary

On species windows the BirdNET target no longer carries a background share: it is BirdNET's opinion renormalised over
the project's species (research R1); background windows keep 003's target. A developer option reproduces 003's target
for comparison. `tools/compare_birdnet.py` runs 8 variants × 4 recording folds of the 006 bundle × 3 seeds with the
normal pipeline (prepare, train, evaluate), and writes the comparison with paired bootstrap intervals; the documents
state its conclusion.

## Technical Context

Python 3.11, existing environments; no new dependency. Tests: `tests/test_birdnet.py` (target), the existing checks
(option off unchanged). The comparison runs on demand (~2 h), not in CI.

## Constitution Check

| Principle | Effect | Status |
|-----------|--------|--------|
| I | the new option is for developers; user text in plain words | Pass |
| II | option off: unchanged; option on: still one command | Pass |
| VI | conclusions only where intervals exclude zero; otherwise "no measurable difference" | Pass |
| Gates | model selection and gates unchanged (folder labels) | Pass |

## Project Structure

```text
wren/torchside/birdnet_target.py   # target(..., y, background_on_species)
wren/torchside/train.py            # passes y and the option; report field unchanged
wren/config.py                     # training.birdnet_background_on_species (default false)
tests/test_birdnet.py              # corrected target tests
tools/compare_birdnet.py           # folds, 96 runs, statistics -> docs/birdnet-comparison.{json,md}
docs/PIPELINE.md, README.md, specs/003-birdnet-distillation/ (pointer)   # the conclusion
```
