---

description: "Task list for feature 005: drop unused packages from the torch environment"
---

# Tasks: Drop unused packages from the torch environment

**Input**: `specs/005-drop-unused-deps/` · **Tests**: the existing checks (FR-004); no new test needed

## Phase 1: User Story 1 — install only what the pipeline uses (P1) 🎯

- [ ] T001 [US1] `envs/lock.sh`: before `uv pip compile`, copy the current `lock/<env>-<plat>.txt` (if any) to the
  `.tmp` output so uv keeps its pins (research R1); document it in `envs/README.md` ("Changing a dependency")
- [ ] T002 [US1] `envs/requirements-torch.in`: remove `torchsummary`, `omegaconf`, `tqdm`; fix its comment if it lists them
- [ ] T003 [US1] `wrennet/models/legacy_modules.py`: remove `from torchsummary import summary` and
  `from omegaconf import DictConfig`; the two `cfg: DictConfig` parameters become `cfg` with a comment "dict-like
  config"; nothing else changes
- [ ] T004 [US1] Run `envs/lock.sh`; check: torch locks lose exactly the 4 packages (research R1), no other pin changes in
  any lock, tf locks identical except their header if any
- [ ] T005 [US1] `python -m wren setup` (reinstalls the torch environment from the new lock); `pip list` shows none of
  the 4 packages
- [ ] T006 [US1] Checks: `tests/test_training.py`, pytest unit checks, synthetic end to end (same `train.json` /
  `history.csv` as `main`), `tests/test_birds_reproduction.sh` and `tests/test_bird_accuracy.sh` as on `main`
- [ ] T007 Push, open a pull request (not merged), confirm CI (Docker: bird reproduction and accuracy byte-exact)
