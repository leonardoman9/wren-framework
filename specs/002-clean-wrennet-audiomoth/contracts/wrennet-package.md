# Contract: the wrennet package

## Layout

```text
wrennet/
├── README.md
├── __init__.py
├── checkpoints/birds.pt
├── models/__init__.py, core.py, legacy_models.py, legacy_modules.py
└── spectral/__init__.py, differentiable_spec.py
```

Nothing else is tracked under `wrennet/` (checked by `tests/test_layout.py`).

## Import

- With the repository root on `sys.path` (the pipeline's `PYTHONPATH`, or `wren/torchside/model.py`'s own insertion):
  `from wrennet.models import Improved_Phi_GRU_ATT` and `from wrennet.models.legacy_modules import CausalConv1d`.
- No environment variable (`WRENNET_ROOT` is gone) and no `sys.path` entry pointing inside `wrennet/`.
- Importing `wrennet` needs only the torch environment's locked packages.

## Invariants

- The six model source files are byte-identical to `wrennet/birds_distillation_edge/{models,spectral}/…` at 7a60637.
- `wrennet/checkpoints/birds.pt` is byte-identical to `wrennet/best_distillation_model.pt` at 7a60637.
- `wrennet/README.md` names the origin (`sciapponi/WrenNet` at 47c7059), the licence (the repository's MIT licence) and
  where the removed training and distillation code lives.
