# Research: Fix the BirdNET target and measure whether it helps

## R1. The corrected target

- **Decision**: `target(confidence, negative, K, T, y, background_on_species=False)`: for windows whose label is a
  species, p = the project species' confidences at their class indices and FLOOR at the background index, normalised,
  softened by T; for background windows, p as in 003. `background_on_species=True` reproduces 003 exactly (for the
  comparison). The mask (skip when every project species < 0.1) is unchanged.
- **Why renormalise rather than skip**: when BirdNET hears the folder's species at 0.3 and an outside species at 0.9,
  its opinion is still informative about the project's species (0.3 versus the others); skipping would discard it.
- **Training option**: `training.birdnet_background_on_species` (default false), documented as a developer option for
  reproducing feature 003's target; plain-language text in README only says "kept for comparison".

## R2. Folds

- The 006 bundle has 4–15 recordings per species and 38 non-bird sounds. 4 folds: recordings of each class sorted by id,
  shuffled with seed 0, dealt round-robin; each fold has every class (the smallest classes have exactly 4).
- Per fold k: test = fold k, train = the other 3 (prepare carves validation from them by recording group,
  `data.group: subfolder`, `split.validation: 0.15`). Implemented with folders of symbolic links
  (`runs/birdnet-compare/folds/k/{train,test}/<class>/XC…`), no audio copied.

## R3. Measures

- From each run's `evaluate/evaluate.json` (float model on the held-out fold's call windows): `call_windows.macro_f1`
  and the false-detection rate = 1 − recall of the non-bird class. Float only: feature 006 measured the device version
  within 1.2 points of float.
- A run whose evaluate stops at the chance gate still writes `evaluate.json`; its figures are kept and marked.
- Per variant: mean over 4 folds × 3 seeds, 95 % bootstrap interval over the 12 runs; paired difference to "without
  BirdNET" with the same init, fold and seed, 95 % bootstrap interval of the paired differences (2,000 resamples,
  seed 0).

## R4. Cost

~60 s per training on macOS arm64 (mps) for these sizes (feature 003: 0.7–1.0 min at 40 epochs), prepare ~5 s per fold,
BirdNET ~5 s per fold and seed set (cached by content across seeds and variants of a fold): 96 runs ≈ 1.5–2 h. Runs
are resumable (each run's folder is skipped when its `train.json` and `evaluate.json` exist).
