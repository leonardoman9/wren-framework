---

description: "Task list for feature 002: clean up wrennet/ and audiomoth/"
---

# Tasks: Clean up wrennet/ and audiomoth/

**Input**: Design documents from `specs/002-clean-wrennet-audiomoth/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: one new check, `tests/test_layout.py` (plan.md, SC-003/SC-005); every other check is an existing one of
feature 001 and is re-run, not rewritten.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [X] T001 Record the baseline on `main` (7a60637): run `tests/test_birds_reproduction.sh` natively and keep its output
  (7 × `identical`, the firmware hashes printed for information), and record the sha256 of
  `wrennet/best_distillation_model.pt` and of the six model source files under
  `wrennet/birds_distillation_edge/{models,spectral}/` (data-model.md identity checks)

---

## Phase 2: Foundational (blocks every story)

- [X] T002 Move the model package with `git mv`, content unchanged (FR-002):
  `wrennet/birds_distillation_edge/models/{__init__,core,legacy_models,legacy_modules}.py` → `wrennet/models/`,
  `wrennet/birds_distillation_edge/spectral/{__init__,differentiable_spec}.py` → `wrennet/spectral/`; and the
  checkpoint `wrennet/best_distillation_model.pt` → `wrennet/checkpoints/birds.pt` (FR-003)
- [X] T003 Create `wrennet/__init__.py`: a one-line docstring naming the package (WrenNet architecture and learned
  front-end) and no imports (contracts/wrennet-package.md)
- [X] T004 In `wren/torchside/model.py`, replace the `WRENNET_ROOT` lookup and `sys.path.insert(… "wrennet")` with an
  insertion of the repository root (`Path(__file__).resolve().parents[2]`) when it is not already on `sys.path`, and
  import `from wrennet.models import Improved_Phi_GRU_ATT`; update the module docstring (FR-004)
- [X] T005 In `wren/torchside/export_backbone.py`, import `CausalConv1d` from `wrennet.models.legacy_modules`
- [X] T006 In `wren/runner.py`, remove `WRENNET_ROOT=…` from the step environment (`PYTHONPATH` already holds the
  repository root) (FR-004)
- [X] T007 Point every checkpoint reference at `wrennet/checkpoints/birds.pt`: `wren/stages/data.py` (train spec and
  key), `examples/birds/project.yaml`, `tests/test_stage_keys.py`, `tests/test_training.py` (FR-003)

**Checkpoint**: the pipeline imports the moved package; nothing else removed yet.

---

## Phase 3: User Story 1 — every check still works (P1) 🎯 MVP

**Goal**: the move changes no output byte.

**Independent Test**: quickstart.md steps 2–3.

- [X] T008 [US1] Verify the identity checks of data-model.md: the six moved source files and `wrennet/checkpoints/birds.pt`
  have the sha256 recorded in T001
- [X] T009 [US1] Run the unit checks (quickstart.md step 2, without `tests/test_layout.py`) and
  `tests/test_birds_reproduction.sh`; the 7 package files must be `identical` as in T001. Stop and investigate if not
  (research.md R2: module paths must not reach the exported files)

**Checkpoint**: commit "wrennet: importable package and checkpoint, moved unchanged".

---

## Phase 4: User Story 2 — only code that is used in wrennet/ (P1)

**Goal**: `wrennet/` = package + checkpoint + README (FR-001).

**Independent Test**: `git ls-files wrennet` lists only the files of contracts/wrennet-package.md.

- [X] T010 [US2] `git rm -r` everything else under `wrennet/` (research.md R1): `birds_distillation_edge/` (the rest of
  it, including `models/export_wrappers.py` and `convert-and-quantize/`), `distillation/`, `datasets/`,
  `dataset_transfer/`, `config/`, `tests/`, `soft_labels_complete/`, `test_soft_labels/`, `train.py`,
  `train_distillation.py`, `extract_soft_labels.py`, `generate_no_birds_samples.py`, `test_birdnet_analyzer.py`,
  `models.py`, `modules.py`, `differentiable_spec_torch.py`, `requirements.txt`, `.python-version`, `.gitignore`,
  `lightning.md`, `directory.md`; then remove the untracked leftovers (`__pycache__/`) from the working tree
- [X] T011 [US2] Rewrite `wrennet/README.md`: what the package holds (architecture, learned front-end, the bird
  checkpoint: 8 species + non-bird), how the pipeline uses it (`init: wrennet`, `examples/birds`), origin
  (`sciapponi/WrenNet` at 47c7059, imported without history), the removed training and distillation code (in that
  repository, and in this one's history at 7a60637), licence (the repository's MIT licence)
- [X] T012 [US2] Re-run `tests/test_training.py` and the pytest unit checks (nothing imported a removed file)

**Checkpoint**: commit "wrennet: drop WrenNet's original training code".

---

## Phase 5: User Story 3 — firmware scripts without WRENNET_ROOT (P2)

**Goal**: contracts/firmware-model-scripts.md.

**Independent Test**: package stage outputs byte-identical; usage lines without `WRENNET_ROOT`.

- [X] T013 [P] [US3] `audiomoth/firmware/bird/gen_bird_tail.py`: drop the first argument; replace
  `sys.path.insert(0, root/"tools")` + `from bird_model_export_check import load_model` with the repository root
  (`parents[3]`) on `sys.path` and `from wren.torchside.model import load_model`; update the usage docstring (also the
  stale `docs/deployment/bird/…` and `tools/bird_model_export_check.py` mentions: the reference is the export stage's
  `reference.npz`)
- [X] T014 [P] [US3] `audiomoth/firmware/bird/e2e_make_ref.py`: the same (imports `SR, load_model`)
- [X] T015 [P] [US3] `audiomoth/firmware/bird/check_frontend.py`: the same (imports `SR, frontend, load_model`)
- [X] T016 [P] [US3] `audiomoth/tests/sim/verify_log.py`: the same with `parents[3]` (`audiomoth/tests/sim/` → repo)
  (imports `load_model`)
- [X] T017 [US3] `wren/stages/package.py`: call `gen_bird_tail.py` and `e2e_make_ref.py` directly with the torch Python
  (`r.torch(name, script, …)`), without `-m wren.torchside.run` and without `ROOT / "wrennet"`
- [X] T018 [US3] Docstrings: `wren/torchside/run.py` (scripts put their own folder on `sys.path`; no WRENNET_ROOT) and
  `wren/torchside/bird_model_export_check.py` (now used by `export_backbone.py` and `make_specs.py` only)
- [X] T019 [US3] Usage in docs: `audiomoth/README.md` (verify_log command), `audiomoth/firmware/app/README.md`
  (verify_log), `audiomoth/firmware/bird/model/README.md` (gen_bird_tail, checkpoint path)
- [X] T020 [US3] Re-run `tests/test_birds_reproduction.sh`: 7 × `identical`

**Checkpoint**: commit "Firmware model scripts: no WRENNET_ROOT argument".

---

## Phase 6: User Story 4 — a smaller audiomoth/ (P3)

**Goal**: research.md R5 table.

**Independent Test**: none of the removed files tracked; firmware tests pass inside the reproduction run.

- [X] T021 [P] [US4] `git rm` `audiomoth/docs/datasheets/AudioMoth_Dev_1_0_1_Datasheet.pdf`, `audiomoth/docs/images/image.png`,
  `audiomoth/firmware/snippets/gpio_cycle_marker.c`, `audiomoth/tools/bird_frontend_v1_reference.c.txt`,
  `audiomoth/firmware/bird/host_test.c`, `audiomoth/firmware/bird/map_sizes.py`, `audiomoth/firmware/bird/sweep_tail.py`,
  `audiomoth/tests/sim/check_wav.py`
- [X] T022 [P] [US4] `git rm` the 63 part headers `audiomoth/vendor/audiomoth/device/inc/efm32wg*f*.h` other than
  `efm32wg380f256.h` (keep `efm32wg_*.h`, `em_device.h`, `system_efm32wg.h`)
- [X] T023 [US4] `audiomoth/README.md`: layout lines for `docs/` (no datasheets; link to the published AudioMoth Dev
  datasheet instead) and `vendor/audiomoth` (only the EFM32WG380F256 part header is kept)
- [X] T024 [US4] Re-run `tests/test_birds_reproduction.sh` (its firmware stage builds the firmware and runs every
  firmware test): PASS

**Checkpoint**: commit "audiomoth: drop unused files and part headers".

---

## Phase 7: Polish & cross-cutting

- [X] T025 Create `tests/test_layout.py` (pytest, ROOT-relative, `git ls-files`): `wrennet/` tracks exactly the files of
  contracts/wrennet-package.md; no tracked file outside `specs/001-reproducible-from-zero/`, `specs/002-clean-wrennet-audiomoth/` and `tests/test_layout.py`
  contains `WRENNET_ROOT`, `birds_distillation_edge` or `best_distillation_model.pt` (text files only); none of the
  T021 files is tracked; exactly one `efm32wg<part>.h` header is tracked
- [X] T026 Add `tests/test_layout.py` to the docker job's unit checks in `.github/workflows/ci.yml` and a row for it to
  the test table in `docs/PIPELINE.md`
- [X] T027 [P] Update `README.md` (layout, checkpoint path in the convert example and licence section) and
  `docs/PIPELINE.md` (model code row, layout line `wrennet/`) (FR-010)
- [X] T028 [P] Update `audiomoth/docs/VERIFICATION.md` mentions of the old checkpoint path, removed files or WrenNet's
  `tools/` where they describe how to run things now (results stay as measured)
- [X] T029 Run quickstart.md steps 1–4 (synthetic end to end included); fix what fails
- [X] T030 Push the branch and open a pull request (not merged: the maintainer reviews the decisions marked in
  spec.md Clarifications); confirm the three CI jobs pass (SC-002), including the Docker byte-exact reproduction

---

## Dependencies & Execution Order

- T001 before anything (baseline from `main`).
- Phase 2 (T002–T007) blocks all stories.
- US1 (T008–T009) verifies Phase 2 and must pass before removals.
- US2, US3, US4 are independent of each other after US1; done in order for one commit each.
- Polish after all stories; T030 last.

## Parallel Opportunities

- T013–T016 (different scripts); T021–T022; T027–T028.

## Implementation Strategy

MVP = Phases 1–3 (the package move with byte-identical outputs). Then each story is one commit verified by the
reproduction test, so a failure points at one change.
