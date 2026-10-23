# Implementation Plan: Learning from BirdNET for bird projects

**Branch**: `003-birdnet-distillation` | **Date**: 2026-10-23 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/003-birdnet-distillation/spec.md`

## Summary

A training option `training: birdnet: true` makes the train stage ask BirdNET v2.4 what it hears in the training audio
(3 s windows on prepare's grid, resampled to 48 kHz) and train WrenNet with the original project's focal distillation
loss, (1 − α)·focal + α·T²·KL, against a corrected teacher target: BirdNET's confidences for the project's species plus
a "none of them" share, normalised and softened by T (research R6). BirdNET is the official v2.4 TFLite model from
Zenodo, fetched by `python -m wren setup --birdnet` and run with `ai_edge_litert`, already in the locked tf environment:
no new package and no lock change. The project check refuses the option when a sound is not a BirdNET species. The
report states BirdNET's version, licence (CC BY-NC 4.0), α, T, γ and how far BirdNET agreed with the folders.

## Technical Context

**Language/Version**: Python 3.11 (torch and tf environments, locked; unchanged)

**Primary Dependencies**: existing locks only: `ai_edge_litert` 2.3.0 (tf env) runs BirdNET; `torchaudio` (torch env)
resamples; setup uses the standard library (`urllib`, `zipfile`, `hashlib`)

**Storage**: `third_party/birdnet/v2.4/{audio-model.tflite,labels/en_us.txt,.wren-birdnet}` (downloaded, gitignored);
per run `runs/<name>/train/birdnet/{windows.f32,grid.json,teacher.npz,stage.json}`

**Testing**: new `tests/test_birdnet.py` (pytest: target, species matching, project-check refusals, loss; no download);
`tests/make_bird_dataset.py` + `tests/birds_birdnet/project.yaml` (end to end with the option on, Docker CI job);
existing checks unchanged (option off: training identical, SC-005)

**Target Platform**: as feature 001 (Docker linux/amd64; native Linux x86_64, macOS arm64)

**Project Type**: CLI pipeline (single repository)

**Performance Goals**: BirdNET adds about a minute or two per hour of training audio (research R4); cached between runs

**Constraints**: constitution I (no "distillation" in user-facing text), II (option off by default), V (bird
reproduction untouched: convert projects never reach the new code), the stage list is unchanged (R8)

**Scale/Scope**: ~8 files edited (`wren/config.py`, `wren/setup.py`, `wren/stages/data.py`, `wren/stages/report.py`,
`wren/torchside/train.py`, `wren/runner.py` labels, `docker/Dockerfile`, `.gitignore`/`.dockerignore`), 4 new modules
(`wren/birdnet.py` stdlib helpers, `wren/torchside/birdnet_windows.py`, `wren/torchside/birdnet_target.py`,
`wren/tfside/birdnet.py`), 3 test files, README,
`docs/PIPELINE.md`, CI

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Effect | Status |
|-----------|--------|--------|
| I. Biologist-first | YAML key `birdnet`, messages and report say "learn from BirdNET"; "distillation" only in `docs/PIPELINE.md` and logs. Species errors list the folders and the fix. | Pass |
| II. One YAML, one command | Off by default; one key turns it on; one extra setup flag, named in the error if missing. | Pass |
| III. Fixed architecture | Unchanged: only the training loss changes. | Pass |
| IV. End-to-end verification | Every check and gate unchanged; model selection and gates on folder labels (FR-005). | Pass |
| V. Reproducibility | Training is already judged by gates, not bytes; BirdNET's results are stored with the run so training can repeat without BirdNET. Convert projects (bird reproduction) never run the new code. The model file is pinned by sha256. | Pass |
| VI. Honesty in reporting | Report: BirdNET version, licence, α/T/γ, agreement per species, suspicious background windows; SC-003 comparison measured, not promised. | Pass |
| Pipeline & Quality Gates | Stage list unchanged; the BirdNET step is part of train with its own cached key. | Pass |

Post-design re-check: pass (no change).

## Project Structure

### Documentation (this feature)

```text
specs/003-birdnet-distillation/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── project-yaml.md
│   ├── setup-birdnet.md
│   └── teacher-target.md
└── tasks.md
```

### Source Code (repository root)

```text
wren/birdnet.py                  # new, stdlib: paths, labels, species matching (used by config.py and setup.py)
wren/setup.py                    # --birdnet: download, sha256, extract, marker; --check row
wren/config.py                   # training.birdnet, birdnet_weight, birdnet_temperature, birdnet_species; checks
wren/stages/data.py              # train: birdnet steps + key; spec passes the teacher file to train.py
wren/runner.py                   # plain labels for the new steps
wren/torchside/birdnet_windows.py  # new: prepared training audio -> 48 kHz windows on the grid
wren/tfside/birdnet.py           # new: BirdNET (ai_edge_litert) -> teacher.npz (project-species confidences, top species)
wren/torchside/birdnet_target.py  # new: target (R6) and focal distillation loss (R7), unit-tested
wren/torchside/train.py          # uses them; birdnet block in train.json
wren/stages/report.py            # plain summary line, technical rows, licence
docker/Dockerfile                # RUN python -m wren setup --birdnet (own layer)
.gitignore, .dockerignore        # third_party/
tests/test_birdnet.py            # new
tests/make_bird_dataset.py       # new: bird project from the calibration bundle
tests/birds_birdnet/project.yaml, project_off.yaml   # new
tests/synthetic/project_birdnet.yaml   # new: the refusal case (non-bird sounds)
.github/workflows/ci.yml         # unit checks + Docker end to end with the option on
README.md, docs/PIPELINE.md      # option, setup flag, licence, method
```

**Structure Decision**: follows the existing split: stdlib code in `wren/` (runs before environments exist), torch
work in `wren/torchside/`, TFLite work in `wren/tfside/`.

## Complexity Tracking

No constitution violations.
