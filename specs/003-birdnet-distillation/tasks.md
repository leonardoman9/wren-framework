---

description: "Task list for feature 003: learning from BirdNET for bird projects"
---

# Tasks: Learning from BirdNET for bird projects

**Input**: Design documents from `specs/003-birdnet-distillation/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: requested by the spec (FR-009, FR-004a): `tests/test_birdnet.py` (no download), the refusal project and the
bird end-to-end project. Test tasks come before the code they check.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [X] T001 Add `third_party/` to `.gitignore` and `.dockerignore` (research R3); and, before any code change, run the
  synthetic project on `main`'s code and save `runs/test-synthetic/train/{train.json,history.csv}` to the scratchpad as
  the SC-005 baseline used by T025
- [X] T002 Create `wren/birdnet.py` (standard library only; imported by `wren/setup.py` and `wren/config.py`): constants
  `DIR = ROOT / "third_party" / "birdnet" / "v2.4"`, `ZIP_URL`, `ZIP_SHA256`, `FILES = {"audio-model.tflite": sha256,
  "labels/en_us.txt": sha256}` (values from data-model.md), `MODEL_NAME = "BirdNET v2.4 GLOBAL 6K (Zenodo 15050749)"`,
  `LICENCE = "CC BY-NC 4.0"`, `CITATION = "Kahl et al. (2021), Ecological Informatics 61:101236"`; `ready()` (the marker
  `.wren-birdnet` lists both sha256), `labels(path=None)` (scientific names in output order, the part before `_`),
  `match(folders, negative, mapping, labels)` → `(indices, unmatched)` per research R5: scientific name =
  `mapping[folder]` else folder with `_` → space; equal ignoring case to exactly one label; background excluded;
  `stored_mapping(run_dir)` → the `species` dict of `runs/<name>/train/birdnet/mapping.json` or `None`
  (data-model.md)

---

## Phase 2: Foundational (blocks both stories)

- [X] T003 `wren/setup.py`: `install_birdnet()` per contracts/setup-birdnet.md: skip if `birdnet.ready()`; else download
  `ZIP_URL` with `urllib` to a temporary file in `third_party/birdnet/`, check `ZIP_SHA256`, extract only the two
  files, check their sha256, write the marker last, delete the zip; on any failure remove partial files, no marker,
  plain message, non-zero exit; print the licence and citation lines. `main(..., birdnet=False)` calls it; `report()`
  adds the row "Learning from BirdNET (optional)" (ready / "run python -m wren setup --birdnet"), not required for
  "Ready"
- [X] T004 `wren/cli.py`: `--birdnet` flag on `setup`, passed to `wren_setup.main`; usage line updated
- [X] T005 `wren/config.py`: `TRAINING_DEFAULTS` gains `"birdnet": False, "birdnet_weight": 0.5, "birdnet_temperature":
  4.0, "birdnet_species": {}` (so `KNOWN["training"]` accepts them); in a convert project any `training.birdnet*` key is
  an error "training.birdnet: only for training projects (data:)"

**Checkpoint**: `python -m wren setup --birdnet` installs BirdNET; projects accept the new keys.

---

## Phase 3: User Story 2 — a clear refusal when the sounds are not birds BirdNET knows (P1)

**Goal**: contracts/project-yaml.md, project check table.

**Independent Test**: `python -m wren run tests/synthetic/project_birdnet.yaml` stops within seconds naming the three
synthetic sounds.

- [X] T006 [P] [US2] Tests in `tests/test_birdnet.py` (pytest, a 12-line fake labels file in `tmp_path`, `birdnet.DIR`
  monkeypatched; no download): `match` accepts `Bubo_bubo`, `bubo bubo`, a mapping `{tawny_owl: Strix aluco}`, and
  reports `upsweep`, `trill` as unmatched; the background folder is never matched; `config.load` on a synthetic-style
  project with `birdnet: true` fails listing every unmatched sound and the `birdnet_species` hint; missing BirdNET →
  the setup message; `loss: ce` + `birdnet: true` → the focal message; `birdnet_weight: 1.5` and
  `birdnet_temperature: 0` → their messages; `birdnet_species` naming a non-folder → its message; `birdnet: false`
  with no BirdNET installed → no error and no BirdNET mention; BirdNET not installed but a `mapping.json` covering every
  sound folder in the run folder → accepted (indices from the mapping); a `mapping.json` missing one folder → "stored
  results do not cover" message
- [X] T007 [US2] `wren/config.py`: with `training.birdnet` true, run the checks of contracts/project-yaml.md in the
  project check (before any stage), in this order: `loss`, `birdnet_weight` in [0, 1], `birdnet_temperature` > 0,
  `birdnet_species` keys are sound folders; then, if `birdnet.ready()`, `birdnet.match` on the class folders with the
  labels file; else, if `birdnet.stored_mapping(p.run_dir)` exists, take the indices from it (every sound folder must be
  in it with the same scientific name, else the "stored results do not cover" message); else the setup message;
  messages exactly as in the contract (plain words, no "distillation"); store the matched label indices in
  `p.training["birdnet_indices"]` for the train stage; default `loss` becomes `focal` when `birdnet` is true and `loss`
  was not given
- [X] T008 [P] [US2] `tests/synthetic/project_birdnet.yaml`: the synthetic project (`tests/synthetic/project.yaml`) with
  `training: {birdnet: true}` and name `test-synthetic-birdnet`, with a comment that it must be refused (US2)
- [X] T009 [US2] Run `pytest tests/test_birdnet.py` and `python -m wren run tests/synthetic/project_birdnet.yaml` (must
  stop in the project check, in under 30 s, naming `downsweep`, `trill`, `upsweep`: SC-001)

**Checkpoint**: commit "Learning from BirdNET: setup and the bird-species check".

---

## Phase 4: User Story 1 — train a bird model that also learns from BirdNET (P1) 🎯 MVP

**Goal**: research R4, R6–R9; contracts/teacher-target.md.

**Independent Test**: quickstart.md steps 4–6.

- [X] T010 [P] [US1] Tests in `tests/test_birdnet.py` for contracts/teacher-target.md, importing the target and loss
  functions from `wren/torchside/birdnet_target.py`: 8 species + background, c = [0.9, 0, …], T = 4 → q[0] ≥ 3/9;
  c all 0 → argmax q is the background; q sums to 1 with no zero for T in {0.5, 1, 4, 10}; no background class →
  p = c normalised; α = 0 equals the focal loss of the same logits, α = 1 equals T²·KL only; for T = 4 the original
  target softmax(c / T) gives q[0] < 1.3/9 (documents the fix); the mask of contracts/teacher-target.md step 4: a batch
  of species windows with every c < 0.1 gives exactly (1 − α)·focal, a background window with c all 0 keeps its KL
  term
- [X] T011 [US1] Create `wren/torchside/birdnet_target.py` (torch only): `target(confidence [B, S], negative, K, T)` →
  q [B, K] per contracts/teacher-target.md steps 1–3 (floor 1e-4); `mask(confidence, y, negative)` per step 4 (species
  window with every confidence < 0.1 → 0); `focal_distillation(logits, y, q, m, alpha, T, gamma)` per step 5 (KL averaged
  over the windows with m = 1), with the focal part identical to `loss_fn` in `wren/torchside/train.py` (move that function here and
  import it back, so both use one definition)
- [X] T012 [P] [US1] Create `wren/torchside/birdnet_windows.py` (`python -m wren.torchside.birdnet_windows PREP_DIR
  HOP_SECONDS OUT_DIR`): for every file of `files_train.json`, 3 s windows of `audio_train.raw` at starts
  `0, h, 2h, … ≤ length − 96000` (h = hop × 32000), int16 → float, resampled 32 → 48 kHz with
  `torchaudio.functional.resample`; implemented as one resampling per file into `audio48.raw` (int16) plus `grid.json`
  with the files (sha256 of the source audio at `files_train.json` `path`, offset, length) and the windows ([file, start])
  — a float32 array of windows would take ~1.4 GB per hour of audio (data-model.md)
- [X] T013 [P] [US1] Create `wren/tfside/birdnet.py` (`python wren/tfside/birdnet.py BIRDNET_DIR GRID_DIR INDICES_JSON
  OUT_DIR`): `ai_edge_litert.interpreter.Interpreter` on `audio-model.tflite`, one window at a time (input [1, 144000]),
  sigmoid of the logits; writes `teacher.npz` with `file_sha256`, `start`, `confidence` (the project species' columns,
  class order), `top_name` (scientific name of the most confident label overall), `top_confidence` (data-model.md);
  prints the count and seconds
- [X] T014 [US1] `wren/stages/data.py` `train()`: when `p.training["birdnet"]`: sub-folder `train/birdnet/` with its own
  key `stage_key("birdnet-1", sorted sha256 of the training files' source audio (from `prepare/files_train.json`
  paths; cached in `train/birdnet/hashes.json` by path, size and mtime), hop, the species mapping, birdnet.FILES)` — no
  paths, times or prepare key in the key (research R8). If up to date,
  reuse it (BirdNET need not be installed: SC-006); else if `birdnet.ready()` run `r.torch("birdnet_windows", …)` then
  `r.tf("birdnet", …)`, write `mapping.json` (data-model.md), delete `audio48.raw`, mark done; else stop before
  training: "BirdNET is needed for the new training audio: run python -m wren setup --birdnet". Add that key and
  `code("torchside/birdnet_target.py")` to the train key; pass `"birdnet_teacher": str(out / "birdnet" /
  "teacher.npz")` and the weight, temperature and gamma in the train spec. With the option off the train key and spec are exactly as before (SC-005)
- [X] T015 [US1] `wren/runner.py` `LABELS`: `birdnet_windows` → "preparing the training audio for BirdNET",
  `birdnet` → "asking BirdNET what it hears in the training audio"
- [X] T016 [US1] `wren/torchside/train.py`: when the spec has `birdnet_teacher`: load `teacher.npz`; map each training
  file (`files_train.json`) to its rows by the sha256 of its source audio, sorted by start (a file with no rows: stop
  with a plain message); in the loop, for each drawn `(fi, start)` (the drawn start, before the jitter inside
  `augmented`) take the row with the nearest start; loss = `focal_distillation(logits, y, target(conf, neg, K, T),
  mask(conf, y, neg), α, T, γ)`; validation loss, `evaluate_split`, model selection and early stopping unchanged
  (FR-005). After training add to `train.json` the `birdnet` block of data-model.md, per class on the listed
  `windows_train.json` windows: BirdNET's top species among all its labels is the folder's / another project species /
  outside the project (with the three most frequent other species by name), the target's largest class is the folder's
  class, windows skipped by the mask; plus background windows with any project species above 0.5; model, licence,
  weight, temperature, gamma, grid size. Update the module docstring (SPEC keys, choices)
- [X] T017 [US1] `wren/stages/report.py`: if `train.json` has `birdnet`, the summary gets "- **Learned from:** your
  folders and BirdNET v2.4 (BirdNET named your folder's species on N % of the training windows); BirdNET's licence is
  CC BY-NC 4.0 (non-commercial), which may also apply to this model" and the technical rows get the model, α, T, γ, a
  per-class table (BirdNET top = folder / other project species / outside the project with the most frequent names,
  target top = folder, skipped windows) and the background count; no "distillation" in the summary
- [X] T018 [P] [US1] Create `tests/make_bird_dataset.py OUT` (standard library + `csv`, copies files): from
  `examples/birds/calibration/ATTRIBUTION.csv`, the 8 model species' clips into `OUT/<Genus_species>/XC<id>/<clip>`,
  at most 10 recordings per species (the 10 lowest xeno-canto ids: SC-003's "10 or fewer", fixed choice), and the
  other species' clips into `OUT/background/XC<id>/<clip>`; refuses to overwrite a non-empty `OUT`
- [X] T019 [P] [US1] `tests/birds_birdnet/project.yaml` (name `test-birds-birdnet`, data root `../../runs/bird_data`,
  negative `background`, `group: file`, `training: {birdnet: true}` (40 epochs, the default: 20 left both versions under-trained, measured in T020), calibration/device as the synthetic
  project) and `project_off.yaml` (same, `birdnet: false`, name `test-birds-plain`), with a comment on the data's
  limits (calibration clips; clips of one recording may fall in different splits)
- [X] T020 [US1] Run quickstart.md steps 1, 4, 5, 5b, 6: setup --birdnet twice (second run downloads nothing), the
  bird project end to end (every stage, firmware tests pass: SC-002), the cache (SC-004: "asking BirdNET" skipped on a
  forced train), training with `third_party/birdnet` moved aside (SC-006), the comparison with the option off (SC-003,
  record both macro-F1; a worse result is reported, not hidden)

**Checkpoint**: commit "Learning from BirdNET: teacher, focal distillation, report".

---

## Phase 5: Polish & cross-cutting

- [X] T021 [P] `README.md`: the `training: birdnet*` options in the project.yaml reference (plain words), a short "Learning
  from BirdNET (birds only)" paragraph: what it does, `python -m wren setup --birdnet`, the licence and citation, the
  species-name rule, and that a copied run folder can be retrained without BirdNET; no "distillation" in the README
  (FR-007)
- [X] T022 [P] `docs/PIPELINE.md`: the train row (focal distillation from BirdNET, corrected target, α/T/γ), the BirdNET
  model and its pins in the pinned-things table, `tests/test_birdnet.py` and the two test projects in the tests table,
  and the SC-003 comparison measured in T020
- [X] T023 `docker/Dockerfile`: `RUN python -m wren setup --birdnet || { tail -50 runs/_setup/logs/setup.log; exit 1; }`
  in its own layer after `--tflm`
- [X] T024 `.github/workflows/ci.yml`: add `tests/test_birdnet.py` to the docker job's unit checks; add a docker step
  "Learning from BirdNET end to end" running `tests/make_bird_dataset.py runs/bird_data` and
  `tests/birds_birdnet/project.yaml`, then the same project again with `third_party/birdnet` moved away inside the
  container and `--stages train --force` (SC-006: must succeed from the stored results); add the refusal run of `tests/synthetic/project_birdnet.yaml` (expects a non-zero
  exit and the three names in the output)
- [X] T025 Option-off check (SC-005): `tests/test_training.py`, the pytest unit checks, the synthetic end to end and
  `tests/test_birds_reproduction.sh` pass as on `main` (package files identical to `main`'s native output); before the
  feature's changes, save `runs/test-synthetic/train/{train.json,history.csv}` from a run on `main` on this computer,
  and check that the branch gives the same best epoch, metrics and `history.csv` (timings excluded)
- [X] T026 Push the branch, open a pull request (not merged), confirm the three CI jobs pass

---

## Dependencies & Execution Order

- T001–T002 → T003–T005 (foundational) → US2 (T006–T009) → US1 (T010–T020) → polish (T021–T026).
- US1 depends on US2's `config.py` work (T007 stores the matched indices the train stage uses).
- Within US1: T010 before T011; T011, T012, T013 before T014 and T016; T018–T019 before T020.

## Parallel Opportunities

- T006 ∥ T008; T012 ∥ T013 ∥ T018 ∥ T019 (different new files); T021 ∥ T022.

## Implementation Strategy

US2 first (small, makes the "birds only" rule real and gives the matched species), then US1 as the MVP of the feature;
each phase is one commit verified by its checkpoint, and the option-off checks (T025) prove nothing else changed.
