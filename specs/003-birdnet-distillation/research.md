# Research: Learning from BirdNET for bird projects

Measured on `main` at f302b9f (feature 002 merged), 2026-10-23.

## R1. Which BirdNET, and how to run it

- **Decision**: BirdNET v2.4 (GLOBAL 6K), the official TFLite release on Zenodo (record 15050749,
  `BirdNET_v2.4_tflite.zip`, 76,822,925 bytes, md5 `c13f7fd28a5f7a3b092cd993087f93f7`, sha256
  `31377e128d86fe7b65fa91b206b8804bad1cd934e624bbe6e7c1788170c95e57`). Only `audio-model.tflite` (sha256
  `55f3e405…65e4`) and `labels/en_us.txt` (sha256 `b50b77b7…7f16`) are kept. It runs with `ai_edge_litert`, which the
  locked tf environment already has (`ai-edge-litert==2.3.0`): **no new Python package, no lock change**.
- **Findings** (run on this machine): input `[1, 144000]` float32 (3 s at 48 kHz), output `[1, 6522]` logits; the
  confidence is `sigmoid(logit)` (BirdNET's default sensitivity 1.0). On one calibration clip each of *Bubo bubo*,
  *Periparus ater*, *Apus apus*, *Emberiza cia* (32 kHz, resampled to 48 kHz), BirdNET's top species is the right one at
  0.90, 0.83, 1.00, 0.93.
- **Alternatives**: the `birdnet` PyPI package 1.3.1 (wraps the same model, downloads it itself; adds dependencies and a
  lock change to the tf environment, whose numpy is pinned at 1.26.4 for the converter); `birdnetlib` (used by the
  original WrenNet code; third-party, needs `tflite-runtime` or TensorFlow and `librosa`); BirdNET+ V3 preview (not a
  stable release). All rejected: the plain model file is what the pipeline needs, and fewer moving parts keep the tf
  environment that the bird reproduction depends on unchanged.

## R2. Licence

- **Decision**: report BirdNET v2.4 as "CC BY-NC 4.0 (Zenodo 15050749)", with the citation Kahl et al. (2021),
  *Ecological Informatics* 61:101236, in the report, the README and `python -m wren setup --birdnet`'s output.
- **Findings**: the Zenodo record's licence field is `cc-by-nc-4.0`; the `birdnet` package README says the models are
  CC BY-NC-SA 4.0 and that research and education count as non-commercial. The primary record wins; the README notes
  that the non-commercial condition may extend to models trained with BirdNET as teacher, without claiming a legal
  conclusion.
- **Distribution**: the model is never committed and never in a published image; `setup --birdnet` downloads it from
  Zenodo on the user's machine (and the locally built Docker image, as for TFLM).

## R3. Where the model files live and how they are fetched

- **Decision**: `python -m wren setup --birdnet` (stdlib only, like the rest of setup) downloads the zip, checks its
  sha256, extracts the two files into `third_party/birdnet/v2.4/` and writes a marker with their sha256. `setup --check`
  reports "Learning from BirdNET: ready / run python -m wren setup --birdnet". `third_party/` is added to `.gitignore`
  and `.dockerignore`; the Dockerfile runs `setup --birdnet` in its own layer (after `--tflm`), so CI and Docker users
  have it.
- **Rationale**: same pattern as `--tflm` (an optional, verified, pinned download); FR-008: nothing else needs it.

## R4. Which audio BirdNET hears

- **Decision**: the prepared training audio (`runs/<name>/prepare/audio_train.raw`, 32 kHz int16, as the model trains
  on), cut into 3 s windows on the grid that prepare uses for species windows (`data.windows.hop_seconds`, 1.5 s by
  default) over every training file (species and background), resampled to 48 kHz
  (`torchaudio`, in the torch environment) and stored as float32 for the tf environment.
- **Why a grid, not only `windows_train.json`**: `wren/torchside/train.py` draws background windows at any position of
  their file and shifts species windows by up to ±0.5 s (augmentation). Each drawn window takes the target of the
  nearest grid position. Species windows start on that same grid (`wren/torchside/prepare.py`: `range(0, length − 3 s,
  hop)`), so they get their own target exactly.
- **Cost**: one BirdNET call per grid step (1.5 s by default) of training audio. Measured order: tens of ms per window on a laptop CPU, so an
  hour of training audio (2,400 windows) takes about a minute or two.

## R5. Matching project sounds to BirdNET species

- **Decision**: for every class except the background, the scientific name is `training.birdnet_species[folder]` if
  given, else the folder name with `_` read as a space; it must equal, ignoring case, the scientific name (the part
  before `_`) of exactly one line of `labels/en_us.txt`. All 8 species of the bird model are there (checked). The check
  runs in `wren/config.py` with the project check (seconds, stdlib), and lists every unmatched sound with the fix
  ("name its scientific name under training: birdnet_species"). If BirdNET is not installed, the stored mapping of a
  previous run is used instead (R8, contracts/project-yaml.md); with neither, the message says to run
  `python -m wren setup --birdnet`.
- **Non-birds**: BirdNET v2.4 has a few non-bird classes (Dog, Engine, Human vocal, …) but no frogs (checked: *Hyla
  arborea*, *Rana temporaria* absent). Only scientific-name species lines are accepted, so "Dog" is not a valid project
  species.

## R6. The distillation target (spec FR-004a)

- **Decision**: for a window, with c_i = sigmoid(logit) for the project's species i (in class order) and the background
  class last: p = [c_1, …, c_S, Π_i (1 − c_i)], floored at 1e-4, normalised to sum 1; the soft target at temperature T
  is q = p^(1/T) / Σ p^(1/T) (the same as softmax(log p / T)). Without a background class (negative: none) p = c,
  normalised.
- **Silent species windows** (spec FR-004a, analysis I1): on a window whose folder label is a species and whose
  confidences for every project species are below 0.1, the KL term is masked out for that window (its weight is 0, the
  focal term keeps weight 1 − α), so a faint call is not pulled towards background. Background windows always use the
  KL term (BirdNET hearing a project species there is information, and is reported). The batch mean of the KL term
  divides by the number of windows that use it.
- **Check** (computed for the spec): a 0.9 detection among 9 classes gives the species 4.95 × (T 3) / 3.92 × (T 4) the
  uniform share; the original softmax(c / T) gave 1.29 × / 1.21 ×.

## R7. The loss (spec FR-004)

- **Decision**: in `wren/torchside/train.py`, with the option on:
  `loss = (1 − α) · mean_focal(logits, y; γ) + α · T² · KL(q ‖ softmax(logits / T))` (KL averaged over the windows
  that use it, R6), defaults α 0.5,
  T 4, γ 2 (`focal_gamma`). This is the original `FocalDistillationLoss` with the corrected target (R6). The focal part
  reuses the existing `loss_fn`. Validation loss, model selection and early stopping stay on cross-entropy and macro-F1
  of the folder labels (FR-005).
- **Interaction with existing options**: the option forces the focal hard loss; `loss: ce` together with the option is
  a project-check error (plain message). `init` wrennet or scratch both work; warm-up (classifier only) unchanged.

## R8. Where in the pipeline (constitution stage list unchanged)

- **Decision**: no new stage. The train stage gets one more step before training, "asking BirdNET about the training
  audio", cached in `runs/<name>/train/birdnet/` with its own key: the sha256 of every training file's source audio
  (content, not path or time), the grid step, the species mapping and BirdNET's files' sha256. The train key includes
  that key, so a change of BirdNET, of the mapping or of the training audio retrains. The files' sha256 are cached in
  `train/birdnet/hashes.json` by path, size and modification time, so an unchanged dataset is not hashed again.
- **Repeating without BirdNET** (spec SC-006, analysis I2): the prepare key holds absolute paths and modification times
  (`wren/runner.py` `listing`), so it changes on another computer; the BirdNET key does not, because it is built from
  file contents. `teacher.npz` stores, per grid row, the source file's sha256 and the start; `mapping.json` stores the
  folder → scientific name → BirdNET index mapping and BirdNET's sha256. On a computer without BirdNET: the project
  check accepts the option when `mapping.json` exists and covers every current sound folder (instead of reading the
  labels file); the train stage reuses `teacher.npz` when its key matches, and otherwise stops before training with
  "BirdNET is needed for the new training audio: run python -m wren setup --birdnet". Prepare must give the same
  training files (same recordings and split seed); it re-runs on the new computer, which is fine because the teacher is
  looked up by file content and start, not by prepare's file index. Steps: `birdnet_windows`
  (torch env: cut and resample) → `birdnet` (tf env: run the model, write `teacher.npz`) → `train` (torch env).
- **Rationale**: the constitution fixes the stage list (prepare → … → report); BirdNET only serves training.

## R9. Report (spec FR-006)

- **Decision**: `train.json` gets a `birdnet` block (model, licence, α, T, γ, windows analysed, and per species on the
  listed training windows: BirdNET's top species among all 6,522 is the folder's species / another project species /
  a species outside the project, with the three most frequent other species; the share whose largest target class is
  the folder's class; the windows with BirdNET's part skipped (R6); plus the background windows BirdNET gives a
  project species above 0.5). `wren/stages/report.py` prints a plain line in the summary ("Learned from BirdNET v2.4 as
  well as from your folders; BirdNET agreed with your folders on 87 % of the training windows") and the rest under
  "Technical details", with the licence line.

## R10. Test data (spec FR-009, SC-002, SC-003)

- **Decision**: `tests/make_bird_dataset.py OUT` builds a bird project from the committed calibration bundle
  (CC0/BY/BY-SA, no BioDCASE): the 8 model species' clips in species folders, one subfolder per recording; the clips
  of the other 62 species as `background` ("none of the project's species"). At most 10 recordings per species (the
  10 lowest xeno-canto ids, so the choice is fixed), matching SC-003's "10 or fewer"; 4–10 recordings and 8–20 clips
  per species. `tests/birds_birdnet/project.yaml` trains on it with the option on; `data.group: file` (4 recordings for
  two species are too few to keep whole recordings in each split), noted as a limit of the comparison.
- **SC-003**: the same project with the option off (`tests/birds_birdnet/project_off.yaml`) gives the comparison,
  recorded in the PR and in `docs/PIPELINE.md`; not a gate.
- **Refusal (SC-001)**: `tests/synthetic` with `birdnet: true` must stop in the project check; a pytest case covers it
  with a small labels file, so it needs no download.
