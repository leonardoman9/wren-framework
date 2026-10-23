---

description: "Task list for feature 004: stages re-run when the model code changes"
---

# Tasks: Stages re-run when the model code changes

**Input**: `specs/004-model-code-stage-keys/` · **Tests**: required by FR-005 (written first)

## Phase 1: Foundational

- [X] T001 `wren/runner.py`: `model_code(root=ROOT)` per research R2 (sorted `wrennet/**/*.py` relative path + bytes,
  then `wren/torchside/model.py`), docstring naming FR-001–FR-003

## Phase 2: User Story 1 — a change to the model code re-runs every stage that depends on it (P1) 🎯 MVP

**Independent Test**: quickstart.md steps 1–2.

- [X] T002 [US1] Tests in `tests/test_stage_keys.py`: `model_code(tmp)` on a copied tree changes after a one-byte edit of
  a `.py` file and not after editing `README.md` or adding a `__pycache__/x.pyc`; `export.input_key`,
  `package.input_key` and `data.train_key` change when `model_code` (monkeypatched in each module) changes; the
  prepare key does not depend on it (FR-004)
- [X] T003 [US1] `wren/stages/export.py`: `model_code()` in `input_key`
- [X] T004 [US1] `wren/stages/package.py`: move the key into `input_key(p, export_dir)` and add `model_code()`
- [X] T005 [US1] `wren/stages/data.py`: move the train key into `train_key(p, prep_key, spec, extra)`; `model_code()`
  instead of `tree_hash(ROOT / "wrennet")` (checkpoint still by content)
- [X] T006 [US1] Run quickstart.md steps 1–3

## Phase 3: Polish

- [X] T007 `docs/PIPELINE.md`: the stage-key test row mentions the model code
- [ ] T008 Push, open a pull request (not merged), confirm CI

## Dependencies

T001 → T002 → T003–T005 (different files, [P] possible) → T006 → T007 → T008.
