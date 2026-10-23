# Research: Stages re-run when the model code changes

## R1. What each stage's key covers (measured on main at f302b9f)

| Stage | loads the model | wren loader (`torchside/model.py`) in key | `wrennet/` in key |
|-------|-----------------|-------------------------------------------|-------------------|
| train | yes | yes (`code`) | whole folder (`tree_hash`), README and checkpoint included |
| evaluate | yes | through train's key | through train's key |
| export | yes | yes (`code`) | no |
| package | yes (`gen_bird_tail.py`, `e2e_make_ref.py`, `windows_ref.py`) | no | no |
| firmware | no | through package's key | through package's key |

## R2. Decision

- **Decision**: `model_code(root=ROOT)` = sha256 over the sorted `wrennet/**/*.py` (relative path + bytes) and
  `wren/torchside/model.py`. Added to export and package keys; replaces `tree_hash(ROOT / "wrennet")` in the train key
  (the checkpoint is already in the train key by content).
- **Why only `.py`**: FR-003 (a README edit re-runs nothing); `__pycache__` holds only `.pyc`.
- **Alternatives**: `tree_hash(wrennet/)` everywhere (re-runs on README edits, hashes the 0.3 MB checkpoint again);
  importing the model and hashing its state (needs torch in the key code, which runs in the pipeline's own Python).
