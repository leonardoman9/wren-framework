# Implementation Plan: Clean up wrennet/ and audiomoth/

**Branch**: `002-clean-wrennet-audiomoth` | **Date**: 2026-10-23 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/002-clean-wrennet-audiomoth/spec.md`

## Summary

`wrennet/` becomes the importable package `wrennet` (`models/`, `spectral/`, moved with their content unchanged) plus
`checkpoints/birds.pt` and a README; WrenNet's original training, distillation, dataset and conversion-experiment code
goes. The pipeline and the firmware's model scripts import the model from the repository (it is on `PYTHONPATH`
already, and the scripts add the repository root themselves), so `WRENNET_ROOT` disappears as an environment
variable and as a script argument. `audiomoth/` drops unreferenced files (datasheet PDF, screenshot, snippets, an old
front-end reference, four developer scripts) and the device headers of the 62 EFM32 variants the AudioMoth does not
use. No conversion code, locked environment or committed reference output changes; the bird reproduction test is the
proof.

## Technical Context

**Language/Version**: Python 3.11 (torch and tf environments, locked); C/C++ (firmware, unchanged)

**Primary Dependencies**: unchanged (envs/lock/*); no lock file is touched

**Storage**: files in the repository

**Testing**: `tests/test_birds_reproduction.sh` (byte-exact package; firmware binaries in Docker), `tests/test_training.py`,
pytest `tests/test_setup.py tests/test_examples.py tests/test_stage_keys.py`, the synthetic end-to-end project, the
firmware tests run by the firmware stage (`audiomoth/scripts/run_tests.sh`), and a new `tests/test_layout.py`
(repository layout and no stale names, SC-003/SC-005)

**Target Platform**: as feature 001 (Docker linux/amd64 reference; native Linux x86_64 and macOS arm64)

**Project Type**: CLI pipeline + embedded firmware (single repository)

**Performance Goals**: none new; the clone gets ~22 MB smaller

**Constraints**: byte-identical bird package and firmware binaries (constitution V); no edits to model source files
(FR-002)

**Scale/Scope**: ~135 files removed from `wrennet/`, ~70 from `audiomoth/`; ~15 files edited

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Effect of this feature | Status |
|-----------|------------------------|--------|
| I. Biologist-first | No user-facing output changes. README paths updated (checkpoint). | Pass |
| II. One YAML, one command | Unchanged; `init: wrennet` keeps its meaning (the bird checkpoint). | Pass |
| III. Fixed architecture | The architecture files move unchanged (FR-002); the state-dict keys and graph are identical. | Pass |
| IV. End-to-end verification | Every parity check and firmware test still runs; developer scripts removed are not part of any check. | Pass |
| V. Reproducibility | Byte-exact bird reproduction must pass in Docker CI (FR-008); locks unchanged. Stage keys change once (hash of `wrennet/` and `audiomoth/`), which only causes one re-run. | Pass |
| VI. Honesty in reporting | Report still records source hashes of `wrennet/` and `audiomoth/`. | Pass |
| Governance | PR checked against principles; no amendment needed. | Pass |

Post-design re-check (after research and contracts): no change, all pass.

## Project Structure

### Documentation (this feature)

```text
specs/002-clean-wrennet-audiomoth/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── wrennet-package.md
│   └── firmware-model-scripts.md
├── checklists/requirements.md
└── tasks.md
```

### Source Code (repository root)

```text
wrennet/                         # after this feature
├── README.md                    # what this is, origin (sciapponi/WrenNet@47c7059), where training code went
├── __init__.py                  # new, minimal
├── models/                      # from birds_distillation_edge/models/: __init__, core, legacy_models, legacy_modules
├── spectral/                    # from birds_distillation_edge/spectral/: __init__, differentiable_spec
└── checkpoints/birds.pt         # was best_distillation_model.pt (same bytes)

wren/torchside/model.py          # imports wrennet from the repository root; no WRENNET_ROOT
wren/torchside/export_backbone.py# CausalConv1d from wrennet.models.legacy_modules
wren/torchside/run.py            # docstring only
wren/torchside/bird_model_export_check.py  # docstring only
wren/runner.py                   # no WRENNET_ROOT in the environment
wren/stages/data.py              # checkpoint path
wren/stages/package.py           # firmware scripts called without the model root
examples/birds/project.yaml      # checkpoint path
tests/test_stage_keys.py, tests/test_training.py  # checkpoint path
tests/test_layout.py             # new
.github/workflows/ci.yml         # runs tests/test_layout.py in the unit checks

audiomoth/firmware/bird/{gen_bird_tail,e2e_make_ref,check_frontend}.py, audiomoth/tests/sim/verify_log.py
                                 # no WRENNET_ROOT argument; model from wren.torchside.model
audiomoth/{README.md,firmware/app/README.md,docs/*.md,firmware/bird/model/README.md}  # paths, removed files
audiomoth/vendor/audiomoth/device/inc/efm32wg*.h  # only efm32wg380f256.h of the part headers stays

README.md, docs/PIPELINE.md      # layout and checkpoint path
```

**Structure Decision**: the repository layout of feature 001 stays; `wrennet/` changes from a copy of another repository
into a small package at the same place, so `tree_hash(ROOT / "wrennet")` and the report keep working unchanged.

## Complexity Tracking

No constitution violations.
