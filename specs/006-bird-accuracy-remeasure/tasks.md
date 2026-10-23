---

description: "Task list for feature 006: re-measure the bird model's accuracy"
---

# Tasks: Re-measure the bird model's accuracy

**Input**: `specs/006-bird-accuracy-remeasure/` (plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md)

**Tests**: required by the spec (FR-006, SC-001, SC-002): statistics on a fixture before the statistics code; the
byte-exact rerun in Docker; the bundle's licences in `tests/test_examples.py`.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [ ] T001 Confirm the keys in `~/.wren-keys` (`XC_API_KEY`, `FREESOUND_API_KEY`) and BirdNET (`python -m wren setup
  --check`); never write the keys into the repository or a log

---

## Phase 2: Foundational (blocks both stories)

- [ ] T002 `wren/torchside/prepare.py`: a split ratio of exactly 0 means no files for that split (no `max(1, …)` carving,
  no "no files in the … split" or "no windows" error for it); ratios above 0 unchanged. Update the module docstring
- [ ] T003 [P] `wren/tfside/device_eval.py`: optional 7th argument `WINDOWS.npz`: write `labels`, `float_logits`,
  `int8_logits` per window (research R2); without it, output exactly as before
- [ ] T004 Run `.venv-torch/bin/python tests/test_training.py` and the synthetic project's `prepare,train` (T002 must not
  change training: same `train.json`/`history.csv` as `main`, timings aside)

**Checkpoint**: commit "prepare: ratio 0 keeps all files in train; device_eval: per-window output".

---

## Phase 3: User Story 1 — honest, current accuracy figures (P1) 🎯 MVP

**Goal**: data-model.md's `accuracy.json` for the shipped package on the new bundle, and the documents.

**Independent Test**: quickstart.md steps 1, 2, 4, 6.

- [ ] T005 [P] [US1] `tools/make_test_bundle.py` per contracts/test-bundle.md (structure of
  `tools/make_calibration_bundle.py`: `argparse`, xeno-canto API v3 with `XC_API_KEY`, `licence()` rules "CC0 / CC BY /
  CC BY-SA only, no NC, ND"): species stretches outside the calibration clips (± 1.5 s), ≤ `--max-seconds 120` per
  recording, "32 kHz mono 16-bit FLAC"; Freesound API v2 search with `FREESOUND_API_KEY`, filter
  `license:"Creative Commons 0"`, HQ previews, first 30 s, BirdNET screening "drop if any scientific-name label is above
  0.1 in any 3 s window" (reusing `wren/tfside/birdnet.py`'s interpreter code through a small helper run in the tf
  environment), `--non-bird 40` balanced over the queries; writes `audio/`, `ATTRIBUTION.csv` (columns of
  data-model.md, `uploaded` from xeno-canto / Freesound `created`), `SHA256SUMS`, `LICENSE.md`, `README.md`
- [ ] T006 [US1] Build the bundle: `. ~/.wren-keys && .venv-torch/bin/python tools/make_test_bundle.py examples/birds/test`;
  record counts per species, recordings per species, unseen recordings (uploaded after 2025-06-11), non-bird sounds and
  total size in research.md; check no file is from BioDCASE and every licence is free
- [ ] T007 [P] [US1] `tests/test_examples.py`: the test bundle's attribution (one row per clip, licences CC0/BY/BY-SA, no
  `-nc`/`-nd`, source xeno-canto or freesound) and `SHA256SUMS` (every listed file matches) — as `test_bundle_attribution`
  does for the calibration bundle
- [ ] T008 [P] [US1] `tests/test_measure.py` (pytest, fixture of ~30 synthetic windows over 6 recordings, no audio):
  bootstrap over recordings is deterministic (seed 0) and its interval contains the point value; windows of one
  recording move together; false-detection rates (any, and ≥ 50 % confidence) on non-bird windows; a class with < 5
  recordings is `"meaningful": false`; the unseen subset is `null` when empty; JSON keys sorted, 4 decimals
- [ ] T009 [US1] `wren/torchside/accuracy.py` (`python -m wren.torchside.accuracy PREP_DIR WINDOWS.npz REF.npz
  ATTRIBUTION.csv CLASSES_JSON OUT_DIR`): recordings from `files_train.json` paths (`XC<id>`/`FS<id>` subfolder) in
  `windows_ref`'s window order, statistics of data-model.md and research R5 (2,000 resamples, seed 0, 95 %), `uploaded`
  > 2025-06-11 subset, writes `accuracy.json` (sorted keys, 4 decimals) and `accuracy.md` (plain tables: overall,
  per species, confusion, false detections, unseen subset, limits)
- [ ] T010 [US1] `tools/measure_bird_accuracy.py` per contracts/measurement.md steps 1–6 (standard library; environments'
  Pythons; logs in `OUT_DIR/logs/`; plain messages); classes from `examples/birds/project.yaml` (`non-bird` last)
- [ ] T011 [US1] Run the measurement; copy `accuracy.json` to `examples/birds/test/accuracy.json`
- [ ] T012 [US1] `audiomoth/docs/VERIFICATION.md` §9: replace the old figures with `accuracy.md`'s (overall, per species
  with counts, float vs device with interval, false detections, unseen subset), the method (bundle, call-finding,
  bootstrap), the limits (FR-008: xeno-canto training overlap except the unseen subset; same recordings as the
  calibration clips at other times; species below 5 recordings not meaningful), the removed analogue-chain rows and why
  (research R7), and the command that reproduces it
- [ ] T013 [P] [US1] `audiomoth/firmware/bird/model/README.md` and `README.md` ("Measured so far"): the new headline
  figures with a link to §9; no old figure presented as current (SC-004)

**Checkpoint**: commit "Bird model accuracy re-measured on a freely licensed test bundle".

---

## Phase 4: User Story 2 — anyone can repeat the measurement (P1)

**Goal**: contracts/measurement.md's reproducibility.

**Independent Test**: quickstart.md step 5.

- [ ] T014 [US2] `tests/test_bird_accuracy.sh`: checks `SHA256SUMS`, runs `tools/measure_bird_accuracy.py`, compares the
  new `accuracy.json` with `examples/birds/test/accuracy.json`: identical in Docker (`WREN_PATH=docker`); natively prints
  the differing figures and passes when all are within 0.002
- [ ] T015 [US2] Run the measurement twice natively: identical outputs (FR-006); run `tests/test_bird_accuracy.sh`
- [ ] T016 [US2] `.github/workflows/ci.yml`: `tests/test_measure.py` in the unit checks; a docker step "Bird model
  accuracy (byte for byte)" running `tests/test_bird_accuracy.sh`. If the Docker result differs from the native one,
  the committed `accuracy.json` becomes the Docker one (constitution V: Docker is the reference) and §9 is regenerated
  from it

---

## Phase 5: Polish

- [ ] T017 [P] `docs/PIPELINE.md`: the measurement in the tests table and a "Bird model accuracy" pointer to §9; the
  BirdNET comparison section names the new bundle as the set for a future comparison
- [ ] T018 Option-off checks: `tests/test_training.py`, the pytest unit checks and `tests/test_birds_reproduction.sh`
  pass as on `main` (package files identical to `main`'s native output)
- [ ] T019 Push, open a pull request (not merged), confirm the three CI jobs pass

---

## Dependencies & Execution Order

- T001 → T002–T004 → US1 (T005–T013) → US2 (T014–T016) → polish (T017–T019).
- In US1: T005 → T006 → T011; T008 → T009; T009 and T010 → T011 → T012, T013.
- US2 needs US1's committed `accuracy.json`.

## Parallel Opportunities

- T002 ∥ T003; T005 ∥ T007 ∥ T008; T012 ∥ T013; T017 with any polish task.

## Implementation Strategy

US1 is the MVP: current, honest figures with their method and limits. US2 then pins them: the same command reproduces
them, byte for byte in Docker, in CI.
