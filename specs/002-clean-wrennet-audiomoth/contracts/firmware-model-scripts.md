# Contract: the firmware's model scripts

Run with the torch environment's Python (`.venv-torch/bin/python`). Each script puts the repository root on `sys.path`
and imports the model loader from `wren.torchside.model`; none takes a model-code path.

| Script | Usage after this feature | Was |
|--------|--------------------------|-----|
| `audiomoth/firmware/bird/gen_bird_tail.py` | `CHECKPOINT.pt REFERENCE.npz OUT_DIR` | `WRENNET_ROOT CHECKPOINT.pt REFERENCE.npz OUT_DIR` |
| `audiomoth/firmware/bird/e2e_make_ref.py` | `CKPT OUT.npz N DIR_OR_FILE...` | `WRENNET_ROOT CKPT OUT.npz N DIR_OR_FILE...` |
| `audiomoth/firmware/bird/check_frontend.py` | `CHECKPOINT.pt LIBFE FILE_OR_DIR [N] [CHUNK]` | `WRENNET_ROOT CHECKPOINT.pt LIBFE FILE_OR_DIR [N] [CHUNK]` |
| `audiomoth/tests/sim/verify_log.py` | `[MODEL_PKG=dir] verify_log.py CKPT WORKDIR` | `[MODEL_PKG=dir] verify_log.py WRENNET_ROOT CKPT WORKDIR` |

## Callers

- `wren/stages/package.py`: `tail_weights` runs `gen_bird_tail.py`, `test_reference` (convert projects) runs
  `e2e_make_ref.py`, both directly with the torch Python (no `wren.torchside.run` wrapper, no model root).
- Documentation: `audiomoth/README.md`, `audiomoth/firmware/app/README.md`, `audiomoth/firmware/bird/model/README.md`.

## Invariant

Outputs are byte-identical to those of the previous call form (same loader functions, same arguments otherwise).
