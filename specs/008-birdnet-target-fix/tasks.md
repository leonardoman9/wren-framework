---

description: "Task list for feature 008: fix the BirdNET target and measure whether it helps"
---

# Tasks: Fix the BirdNET target and measure whether it helps

**Input**: `specs/008-birdnet-target-fix/` · **Tests**: required (FR-003)

## Phase 1: User Story 1 — BirdNET's opinion never contradicts a species label (P1)

- [ ] T001 [US1] Tests in `tests/test_birdnet.py`: the US1 scenario (project species 0.3 and 0.05, outside species
  0.9 → no background share, top class the 0.3 species); a background window equals 003's target;
  `background_on_species=True` equals 003's target on every window; the skip below 0.1 unchanged
- [ ] T002 [US1] `wren/torchside/birdnet_target.py`: `target(confidence, negative, classes, temperature, y=None,
  background_on_species=False)` per research R1; docstring
- [ ] T003 [US1] `wren/torchside/train.py`: pass `y` and `spec["birdnet_background_on_species"]` to `target` in the
  loss and in the report's "target top = folder"
- [ ] T004 [US1] `wren/config.py`: `birdnet_background_on_species` in `BIRDNET_DEFAULTS` (false), boolean check;
  README options list ("developer option: feature 003's target, kept for comparison")
- [ ] T005 [US1] Run `tests/birds_birdnet/project.yaml` `--stages prepare,train`: "target's top class = folder" ≥ 90 %
  for every species BirdNET hears above 0.1 (SC-001); option-off checks unchanged (FR-006)

## Phase 2: User Story 2 — a sound answer to "does BirdNET help?" (P1)

- [ ] T006 [US2] `tools/compare_birdnet.py` (research R2–R4): folds from `examples/birds/test/` (4, round-robin per
  class, seed 0, written to `runs/birdnet-compare/folds.json`), symlinked fold folders, one project YAML per variant ×
  fold × seed, `python -m wren run … --stages prepare,train,evaluate` per run (resumable), statistics with paired
  bootstrap, outputs `docs/birdnet-comparison.json` and `.md`
- [ ] T007 [US2] Run the comparison (≈ 2 h); commit the outputs
- [ ] T008 [US2] `docs/PIPELINE.md` (replace the "Learning from BirdNET: measured so far" section), `README.md` (one
  plain sentence with the conclusion), `specs/003-birdnet-distillation/research.md` (pointer to 008): state only what the
  intervals support (SC-003)

## Phase 3: Polish

- [ ] T009 Unit checks, synthetic end to end, `tests/test_birds_reproduction.sh`, `tests/test_bird_accuracy.sh` as on
  `main`
- [ ] T010 Push, pull request (not merged), CI
