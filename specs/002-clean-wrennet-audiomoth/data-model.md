# Data Model: Clean up wrennet/ and audiomoth/

This feature moves and removes files; it adds no data format. The entities below are the two things whose identity
must survive the move.

## WrenNet model package

| Field | Value |
|-------|-------|
| Import name | `wrennet` (folder `wrennet/` at the repository root) |
| Public names used | `wrennet.models.Improved_Phi_GRU_ATT`, `wrennet.models.legacy_modules.CausalConv1d` |
| Source files | `models/{__init__,core,legacy_models,legacy_modules}.py`, `spectral/{__init__,differentiable_spec}.py` — content identical to `birds_distillation_edge/…` at 7a60637 |
| Changed files | `wrennet/__init__.py` (new, minimal) |
| Identity check | the 6 source files' sha256 equal those of the old paths (FR-002) |

## Bird checkpoint

| Field | Value |
|-------|-------|
| Path | `wrennet/checkpoints/birds.pt` (was `wrennet/best_distillation_model.pt`) |
| Format | PyTorch state dict, 9 outputs |
| Identity check | sha256 unchanged; `audiomoth/firmware/bird/model/package.json`-style `checkpoint_sha256` in the run's package equals the previous one |
| Used by | `examples/birds/project.yaml` (`checkpoint:`), `wren/stages/data.py` (`init: wrennet`), `tests/test_training.py`, `tests/test_stage_keys.py` |
