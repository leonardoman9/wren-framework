# Step 3: from audio folders to a trained checkpoint (prepare, train, evaluate)

Revised after a review (an independent agent read the plan against WrenNet's code and ran checks; its blockers were
confirmed here before acting on them).

## Context
Step 2 converts an existing WrenNet checkpoint into AudioMoth firmware. Step 3 adds training, so a user with audio of
*any* animals gets a model without writing code. Fixed: the WrenNet architecture, 32 kHz, 3 s windows, hidden size 64,
the operator set. Chain: `prepare → train → evaluate → export → package → firmware → report`.

## Decisions
- **Our own training loop, WrenNet's model class.** WrenNet's trainer is tied to its bird project:
  - ESC-50 is always loaded;
  - "no bird" segments are cut from every bird file whatever the split, so they leak between splits;
  - seed 42 is hard-coded for some splits, and `os.listdir` order makes the splits depend on the filesystem;
  - call extraction only looks at 2–10 kHz (owls, ducks and frogs are missed), and only one segment per file is used;
  - it falls back to a synthetic dataset, with only a warning, when files are missing;
  - its dependencies (`numpy<2`) conflict with the verified environment.
- **One model constructor** (`build_model(classes)`), used by training and by export: f_min/f_max = 150/16000. Without
  them a model built from scratch has a NaN filterbank; today only `load_state_dict` hides this.
- **dB floor per clip.** WrenNet's `AmplitudeToDB(top_db=80)` on a batch takes one floor for the whole batch; the C
  front-end takes each clip's own. Measured on 16 real clips: 4 change class between batched and one at a time.
  Training and evaluation use a per-clip front-end (`frontend` + `Core`), so batched = one at a time = device. (The bird
  model was trained with the batch floor; all deployment checks so far used one clip at a time, so they hold.)
- **Learned filterbank parameters.**
  - Fine-tuning (`init: wrennet`, default) freezes breakpoint and transition width. The C tables stay those of the bird
    model, and the run is deterministic.
  - From scratch they are trained with WrenNet's learning rates (3.0 and 0.1; at 1e-4 the breakpoint cannot move
    within float32), without weight decay, with the filterbank cache rebuilt at every step (otherwise gradients are
    lost or the backward fails).
  - Their constraints are checked before export: finite, transition width > 0.
- **Device operating point.**
  - **Batches:** a configurable share of background windows (default 40 %), the other classes equally, every file
    weighted equally within its class.
  - **Loss:** cross-entropy (focal optional), so balance is corrected once.
  - **Classifier bias:** set to the log of the class shares.
  - **Threshold:** evaluate suggests the `device.confidence` threshold.
- **Augmentation** (training only; gain is useless because of the per-clip standardisation):
  - random offset jitter around the selected window, within the file;
  - background mixing at 5–20 dB SNR;
  - ±5 % speed;
  - random low-pass (robustness to bandwidth);
  - SpecAugment on the standardised spectrogram.
  - Not used: WrenNet's Griffin-Lim masking, and the circular roll, which leaves a click at the wrap point.
- **New choices, not WrenNet's:** cosine learning-rate schedule, 2 warm-up epochs with the pretrained backbone frozen,
  and early stopping on validation macro-F1 (tie: validation loss). Fine-tuning loads every weight except `fc`, and
  checks that only `fc` is missing.

## Data
```
data/<class>/**/*.wav|flac|mp3|ogg     one folder per class (any case of extension, sub-folders allowed)
data/<negative>/                        the "nothing of interest" class: site noise, wind, rain, other species
```
```yaml
data:
  root: data/                    # split automatically (stratified by class, by group), or explicit folders:
  # train: data/Train            #   validation is carved from train when not given; the given validation becomes
  # validation: data/Validation  #   the test when there is no test (so model selection never sees the test)
  # test: data/Test
  negative: background           # folder of the negative class (placed last); omit if the model has none
  names: {Common_Chaffinch: "Fringilla coelebs"}   # optional display names for the log (ASCII, <= 40 chars)
  group: file                    # file | subfolder: what must not be split across train/validation/test
  split: {validation: 0.15, test: 0.15, seed: 0}
  windows: {hop_seconds: 1.5, max_per_file: 6, negative_max_per_file: 40}
training: {init: wrennet, epochs: 40, batch_size: 32, lr: 5.0e-4, negative_share: 0.4, loss: ce, augment: true,
           seed: 0, device: auto}
```

## Stages
1. **prepare** (`wren/torchside/prepare.py`)
   - **Scan.** Find files (sorted, any case of extension) and validate class names against the firmware's rules
     (ASCII, no quotes, commas or backslashes, 1–40 characters, unique, 2–180 classes).
   - **Split.** By group, stratified, with a fixed seed. Explicit split folders must have the same classes. Exact
     duplicates (content hash) within or across splits are an error.
   - **Decode.** Every file once, in blocks: mono, 48 Hz DC high-pass as on the device, resampled to 32 kHz, int16.
     Clipping is counted. The result is stored per split as one memory-mapped array plus an index; windows are views
     of it, not copies.
   - **Windows.**
     - **Positive classes:** score each 3 s window by its prominence: dB above each FFT bin's floor over the file
       (its 20th percentile, so that a chorus calling in most frames does not become the floor), the 8 most
       prominent bins per frame, the 30 most prominent frames per window.
       - Only the class's band counts. Per FFT bin (150–16000 Hz), take the share of the class's training 3 s windows
         with an event there (≥ 3 frames 20 dB above the floor). Subtract the same share in the negative class, or,
         without one, the median over the other classes. Noise the reference also has, such as wind, rain or hum,
         cancels out.
       - The band is ±1 octave around the highest contrast, clamped to 150–16000 Hz.
       - The whole range is used instead when the contrast is below 0.05, or when bins more than 2 octaves away reach
         3/4 of it (a warning: events in several bands, or a broadband call).
       - A band below 1 kHz gives a warning. Noise found only in one class's recordings cannot be told apart from the
         animal, so add background recorded at the same sites.
       - Clipped files give a warning: clipping spreads noise over every band.
       - The bands are in `prepare.json` and in the report.
       Keep windows at least 3 s apart, 6 dB above the file's noise reference (20th percentile of the frames), up to
       `max_per_file`. Files of 6 s or longer without an event give no windows and are counted. Shorter clips keep
       their best window: the clip is the label.
     - **Negative class:** the device grid, capped per file.
     - **Short files:** extended by reflection, not with digital silence. The padded share per class is reported.
   - **Report.** `prepare.json`: counts per class and split, duration, sample-rate and bandwidth distribution per class,
     with a warning when classes differ (a shortcut the model could learn), unreadable and clipped files, files
     without events, padding.
   - **Hard errors.**
     - fewer than 5 files in a class;
     - an empty split for a class;
     - bad class names;
     - more than 180 classes;
     - duplicates across splits;
     - unreadable data that would leave a class empty.
2. **train** (`wren/torchside/train.py`)
   - Training follows the decisions above. Augmentation runs in the main process with a seeded generator.
   - On CPU: deterministic algorithms and a fixed thread count. MPS/CUDA results can differ between runs, possibly
     in the best epoch; the spread is reported.
   - **Outputs.**
     - `best.pt`: a plain state_dict, checked to load with `load_model`;
     - `history.csv`;
     - `train.json`.
3. **evaluate** (torch, on the test split)
   - **(a) Call windows** (those prepare selected): accuracy, macro-F1, per-class precision/recall/F1, confusion matrix.
   - **(b) Device decision per file:** windows on the device grid (`device.hop_seconds`) and the device rule (top
     class, softmax ≥ threshold, not the negative class) give "species reported in this file". Recall and precision
     per class, for thresholds from 0.3 to 0.9, and the suggested `device.confidence`.
   - **(c) Negative files:** false reports per hour at that threshold.
   - **Gate:** macro-F1 of (a) must exceed 1/N + 0.1, otherwise it is a pipeline bug or unusable data.
   - **Outputs:** `evaluate.json`, `confusion.csv`.

Then:
- **export** calibrates on training windows drawn from the device grid, stratified by class, and measures the int8
  error on validation windows (new `specs_from_windows.py`; checkpoint projects keep the verbatim WrenNet script,
  for byte-exact reproduction).
- **package** runs the host device pipeline (C front-end, int8 TFLite, C tail) over the test call windows and reports
  float vs int8 metrics. The gate is a macro-F1 drop ≤ 0.05.
- **firmware** and **report** are unchanged; the report gains data, training and evaluation sections.
- **Stage keys** hash the file lists (path, size, modification time) and the code of the stage.

## Verification
1. **Unit checks:**
   - batched logits = one-clip logits;
   - from scratch: finite filterbank and loss;
   - from scratch the breakpoint moves, and the C front-end with the new tables matches PyTorch (the package stage's
     device check);
   - window selection on synthetic long files with known event positions (found, at least 3 s apart, none in pure
     noise);
   - padding;
   - split disjointness and determinism (same result twice, under any directory order);
   - error paths: < 5 files, bad names, an unreadable file, mismatched split folders, duplicates.
2. **Synthetic smoke test** (fast, also for CI): 3 tonal "species" + noise background in 4–8 s files. From scratch it
   must reach ≥ 95 % test accuracy; then the whole chain to the firmware and its tests.
3. **Real data, new classes:** BioDCASE 2026 TinyML development set, 10 European species not among WrenNet's 8, plus
   Background (11 classes).
   - Validation is carved from Train by file; the given Validation is the test.
   - `init: wrennet` vs `scratch`, and a second seed, to judge the spread and justify the default.
   - Then export, package (int8 vs float) and the firmware tests.
   - Limits reported: every file is 3 s at 24 kHz, so this does not exercise long files, and nothing above 12 kHz.
4. **Steps 1–2 unchanged:** `tests/test_birds_reproduction.sh` (byte-exact) and the 4-class checkpoint project still
   pass.
5. **Not covered here, stated in the report:** long field recordings and false positives per hour on real site
   background (no such data on this machine).
