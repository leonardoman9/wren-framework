# Quickstart: validating feature 004

1. `.venv-torch/bin/python -m pytest -q tests/test_stage_keys.py` → the new key tests pass.
2. `tests/test_birds_reproduction.sh` → package files and firmware identical to `main`'s native output (FR-006).
3. Manual (SC-001/SC-002): run `examples/birds/project.yaml` to the end; add a comment line to
   `wrennet/models/core.py` → a second run re-runs export, package, firmware; revert; edit `wrennet/README.md` only →
   every stage "up to date".
