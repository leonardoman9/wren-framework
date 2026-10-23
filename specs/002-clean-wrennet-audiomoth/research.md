# Research: Clean up wrennet/ and audiomoth/

All findings measured on `main` at 7a60637 (2026-10-23).

## R1. What the pipeline uses from wrennet/

- **Decision**: keep `birds_distillation_edge/models/{__init__,core,legacy_models,legacy_modules}.py`,
  `birds_distillation_edge/spectral/{__init__,differentiable_spec}.py` and `best_distillation_model.pt`; remove the rest.
- **Findings**:
  - `wren/torchside/model.py` imports `birds_distillation_edge.models.Improved_Phi_GRU_ATT`;
    `wren/torchside/export_backbone.py` imports `birds_distillation_edge.models.legacy_modules.CausalConv1d`.
    `models/__init__.py` imports `core`, `legacy_models`, `legacy_modules`; `legacy_models` imports
    `..spectral.differentiable_spec`. Nothing else under `wrennet/` is imported (`git grep`).
  - `models/export_wrappers.py` is not imported by `models/__init__.py` nor by wren: removed.
  - The package `__init__.py` reads its version from installed metadata and lazily imports `datamodule` (Lightning):
    replaced by a minimal one.
  - The checkpoint is a plain state dict (tensor keys such as `fc.weight`), so no pickled class path depends on the
    module name `birds_distillation_edge`.
  - The remaining code imports PyTorch Lightning or Hydra (`datamodule.py`, `experiment.py`, `cli/train.py`,
    `cli/export_onnx.py`, `data/dataset_factory.py`, `datasets/dataset_factory.py`), which the locks do not install.
- **Alternatives**: keep it under `wrennet/training/` (rejected: cannot run with the locked environments, and it is
  preserved in `sciapponi/WrenNet` and in git history); trim unused classes from the model files (rejected for this
  feature: FR-002, the computation must be unchanged by construction).

## R2. Package name and import path

- **Decision**: the folder `wrennet/` is the package (`wrennet/__init__.py`), imported as `wrennet.models` from the
  repository root. `wren/runner.py` already puts the repository on `PYTHONPATH` for every step; `wren/torchside/model.py`
  inserts the repository root (`parents[2]`) on `sys.path` for direct use (tests, developer scripts).
- **Rationale**: no environment variable; the import name matches the folder and the architecture's name.
- **Risk checked**: a package named `wrennet` at the root and the old `sys.path.insert(ROOT/"wrennet")` must not both
  exist, or `models` would also be importable as a top-level name. The old insertion goes.
- **Does the module path reach the exported files?** The ONNX graph's node names come from the module attribute
  hierarchy (`phi.…`, `gru`), not from Python module paths, and the exporter keeps no source locations in its output.
  Verified by running `tests/test_birds_reproduction.sh` (R6).

## R3. Checkpoint location and name

- **Decision**: `wrennet/checkpoints/birds.pt` (git rename, identical bytes, sha256 unchanged).
- **References to update**: `wren/stages/data.py` (train spec and key), `examples/birds/project.yaml`,
  `tests/test_stage_keys.py`, `tests/test_training.py`, `README.md`, `docs/PIPELINE.md`, `audiomoth/README.md`,
  `audiomoth/firmware/bird/model/README.md`.
- **Effect on outputs**: the path is not written into any package file (`model_config.json` holds names and device
  settings; `package.json` holds the checkpoint's sha256, not its path). `stage_key` hashes files by content, so a
  convert project's export key does not change with the path; the train key changes (it holds the path string and
  `tree_hash(wrennet/)`), and every key that hashes `audiomoth/` changes. Affected stages re-run once.

## R4. Firmware model scripts without WRENNET_ROOT

- **Decision**: `gen_bird_tail.py`, `e2e_make_ref.py`, `check_frontend.py` and `tests/sim/verify_log.py` drop the first
  argument and import `SR`, `frontend`, `load_model` from `wren.torchside.model`, after putting the repository root
  (`Path(__file__).resolve().parents[N]`) on `sys.path`. The package stage calls the first two directly with the torch
  Python instead of through `python -m wren.torchside.run`.
- **Findings**: today each script does `sys.path.insert(0, root / "tools")` and `from bird_model_export_check import …`.
  `WRENNET_ROOT/tools` does not exist in this repository (WrenNet's `tools/` was never published), so the scripts only
  work through `wren.torchside.run`, which puts wren's stand-in into `sys.modules`. The stand-in re-exports the same
  functions from `wren.torchside.model`, so importing them directly runs the same code.
- **Kept**: `wren.torchside.run` and the stand-in remain for `wren/torchside/export_backbone.py` and `make_specs.py`,
  which are copies of WrenNet's tools and import `bird_model_export_check` themselves (export path of the bird
  reproduction; not edited).
- **Alternatives**: keep the `bird_model_export_check` import with a fallback (rejected: two import paths for one
  function); move the scripts into `wren/` (rejected: they belong to the firmware and its README).

## R5. audiomoth/ files

Every tracked file was checked for a reference (by file name) from a build script, a test script, wren, CI or a
document.

| File | Referenced by | Decision |
|------|---------------|----------|
| `tools/board/*` (4 scripts) | `docs/FIRST_BOARD_TEST.md`, `firmware/app/main_app.c` | keep |
| `tools/energy_parser.py` | `docs/FIRST_BOARD_TEST.md` step 5 | keep |
| `tools/memcount.c` | `README.md`, `docs/VERIFICATION.md`, `firmware/bird/bird_marker.h` | keep |
| `tools/gen_frontend_tables.c` | package stage | keep |
| `firmware/bird/check_frontend.py` | `tools/gen_frontend_tables.c` (how the tables were verified) | keep, no WRENNET_ROOT |
| `docs/datasheets/AudioMoth_Dev_1_0_1_Datasheet.pdf` (2.8 MB) | `README.md` layout line only | remove; link to the published datasheet |
| `docs/images/image.png` (1.4 MB) | nothing | remove |
| `firmware/snippets/gpio_cycle_marker.c` | nothing | remove |
| `tools/bird_frontend_v1_reference.c.txt` | nothing | remove |
| `firmware/bird/host_test.c` | nothing (not built by any script) | remove |
| `firmware/bird/map_sizes.py` | nothing | remove |
| `firmware/bird/sweep_tail.py` | nothing (takes WRENNET_ROOT) | remove |
| `tests/sim/check_wav.py` | nothing (not run by `scripts/run_tests.sh`) | remove |
| `vendor/audiomoth/device/inc/efm32wg<part>.h`, 62 of 63 | `em_device.h` selects one by the `EFM32WG380F256` define (`build_app.sh`, `build_sim.sh`); the others are never included | remove all but `efm32wg380f256.h` (~19 MB) |

`em_device.h` is not edited: its `#elif` chain only includes the header of the defined part, and a build for another
part fails on the missing header, which is the right outcome for firmware that only targets the AudioMoth.

## R6. Verification

- Native (macOS arm64, this machine): `tests/test_birds_reproduction.sh` compares the 7 package files byte for byte; it
  was run on `main` before the change as the baseline, and is run again after.
- Docker (CI): the same test also compares both firmware binaries with `tests/expected/birds_firmware.sha256`.
- The firmware stage runs `audiomoth/scripts/run_tests.sh` (host, simulator, fault, schedule, file-system, QEMU tests),
  so the vendor trim is exercised.
