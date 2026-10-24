# Implementation Plan: Drop unused packages from the torch environment

**Branch**: `005-drop-unused-deps` | **Date**: 2026-10-24 | **Spec**: [spec.md](spec.md)

## Summary

Remove `torchsummary`, `omegaconf` and `tqdm` from `envs/requirements-torch.in`, the two imports from
`wrennet/models/legacy_modules.py` (the two `DictConfig` annotations become plain parameters), and regenerate the four
locks with `envs/lock.sh`, which from now on starts from the current lock so that uv keeps every other version
(research R1). The Docker image's environment layer is rebuilt once; the bird reproduction stays byte-exact.

## Technical Context

**Language/Version**: Python 3.11 · **Tooling**: uv 0.13.4 (`envs/lock.sh`) · **Testing**: `tests/test_training.py`,
pytest unit checks, synthetic end to end, `tests/test_birds_reproduction.sh`, `tests/test_bird_accuracy.sh`, CI ·
**Constraints**: no other locked version changes (FR-003); outputs byte-exact in Docker (FR-004)

## Constitution Check

| Principle | Effect | Status |
|-----------|--------|--------|
| III. Fixed architecture | no executable statement of the model changes; only imports and two annotations | Pass |
| V. Reproducibility | locks regenerated with the project's tool, other pins unchanged; Docker reproduction must stay byte-exact | Pass |
| Governance (envs/README.md "Changing a dependency") | followed: edit `.in`, `envs/lock.sh`, reproduction in Docker | Pass |

## Project Structure

```text
envs/requirements-torch.in            # three lines removed
envs/lock.sh                          # prefer the current lock's pins (copy it to the output before compiling)
envs/lock/torch-*.txt, tf-*.txt       # regenerated; tf locks must come out identical
wrennet/models/legacy_modules.py      # imports and annotations
envs/README.md                        # the "prefer current pins" behaviour
```
