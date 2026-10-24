# Research: Drop unused packages

## R1. Regenerating the locks without moving other versions

- **Finding**: `envs/lock.sh` compiles each lock into a fresh `.tmp` file, so uv resolves every transitive dependency to
  its newest version: a plain regeneration would change many unrelated pins.
- **Decision**: copy the current lock to the output file before compiling: uv prefers the versions already in the
  output file (it upgrades only with `--upgrade`). Dry run on `torch-macos-arm64` (2026-10-24, uv 0.13.4): removed
  exactly `antlr4-python3-runtime`, `omegaconf`, `torchsummary`, `tqdm`; added none; changed none.
- **Also good for later**: any future `.in` change now moves only what it must.

## R2. Uses in the code

`torchsummary.summary` is imported and never called; `omegaconf.DictConfig` only annotates two constructor parameters
(`MatchboxNetSkip`, `ImprovedPhiGRU…` config) and appears in one comment; `tqdm` is imported nowhere in `wren/`,
`wrennet/`, `tests/`, `tools/` or the firmware's scripts. `wren/torchside/model.py` passes plain dictionaries.
