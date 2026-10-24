# Research: Re-measure the bird model's accuracy

Measured on `main` at 7342e87 (features 001–004 merged), 2026-10-23.

## R1. What the old figures were and why they cannot be repeated

`audiomoth/docs/VERIFICATION.md` §9: 161 windows (the 2 loudest per file) of Drive files, scripts
`tools/bird_eval_prepare.py` / `bird_eval_pipeline.py` / `xc_fetch.py` that live in the WrenNet repository, on the
package before feature 001's recalibration. No background windows, no per-recording intervals.

## R2. The pieces already in the pipeline

- `wren/torchside/prepare.py`: decodes folders to 32 kHz, finds call windows per class (FR-005) and background windows
  on a 3 s grid. Its SPEC takes the classes in label order (negative last), so passing the checkpoint's output order
  (`examples/birds/project.yaml`) gives labels in the model's order directly. **Change needed**: today it always carves
  at least one group into validation and test (`max(1, round(ratio × groups))`) and refuses a file in two splits, so it
  cannot keep everything in one split; a ratio of exactly 0 will mean "none" (no carving, no "empty split" error for that
  split). Ratios above 0, i.e. every training project, behave exactly as before (checked by `tests/test_training.py`
  and the synthetic project).
- `wren/torchside/windows_ref.py`: float64 PyTorch logits of labelled windows (pcm, logits, labels) in the firmware's
  reference format.
- `wren/tfside/device_eval.py`: the device pipeline on the computer (C front-end + int8 TFLite with the reference
  kernels + C tail, the firmware's own C) against the float logits; accuracy and macro-F1 for both.
- The shipped package `audiomoth/firmware/bird/model/` (int8 backbone, generated headers) and the checkpoint
  `wrennet/checkpoints/birds.pt`.
- **Decision**: the measurement reuses these; `prepare.py` gets the zero-ratio change above, `device_eval.py` an
  optional output of the per-window results (labels, float and int8 probabilities) so the statistics can be computed
  outside it, and `windows_ref.py` keeps every window when asked for more than there are (it already does: `[:n]`).
  The windows' order is shuffled with a fixed seed in `windows_ref`; the statistics do not depend on order.

## R3. Species test audio (spec FR-002)

- **Decision**: a maintainer tool `tools/make_test_bundle.py` (XC_API_KEY, as `make_calibration_bundle.py`) fetches every
  CC0 / CC BY / CC BY-SA recording of the 8 species (API v3, quality A–C), removes the spans of the calibration clips
  of that recording (`ATTRIBUTION.csv` `start_s` .. +3 s, plus 1.5 s on each side), keeps up to 120 s per recording,
  and writes the remaining stretches of at least 3 s as `examples/birds/test/audio/<Genus_species>/XC<id>/XC<id>_<k>.flac`
  (32 kHz mono 16-bit, like the calibration clips), with `ATTRIBUTION.csv` (clip, xc_id, species, recordist, licence,
  url, source span, `uploaded`), `SHA256SUMS`, `LICENSE.md`, `README.md`.
- **Why stretches, not chosen windows**: the measurement's own call-finding (prepare) picks the windows (FR-005), so the
  method is in the repository, not in the tool.
- **Size**: about 82 recordings, at most 120 s each: at most ~2.7 h, expected ~45 min (~45 MB of FLAC, like the
  calibration bundle's 47 MB).
- **Unseen subset**: WrenNet's checkpoint was first committed to `sciapponi/WrenNet` on 2025-06-11 (df95736), so its
  training data were collected before that date. Recordings with xeno-canto `uploaded` after 2025-06-11 are certainly
  unseen and are reported as a separate subset (FR-002, FR-008).
- **Minimum**: a species with fewer than 5 test recordings is reported but marked "not meaningful" (SC-002). Known
  today: *Emberiza cia* and *Poecile montanus* have 4 free recordings each.

## R4. Non-bird test audio (spec FR-003)

- **Decision**: CC0 field recordings from Freesound (API v2, `license:"Creative Commons 0"`, tags such as wind, rain,
  stream, river, insects, traffic, footsteps, forest ambience without birds), fetched by the same maintainer tool
  (FREESOUND_API_KEY), at most 30 s each, up to ~40 sounds, and **screened by BirdNET** (installed by feature 003): a
  sound is kept only if no 3 s window has any bird species (a BirdNET label with a scientific name) above 0.1. With a
  token key Freesound serves only the **HQ previews** (MP3/OGG, 44.1 kHz), not the original files (OAuth2 login): the
  previews are used, resampled to 32 kHz (wind, rain and streams lose nothing that matters at 128 kbps). Stored as
  `examples/birds/test/audio/non-bird/FS<id>/FS<id>.flac` with attribution (CC0 needs none; kept for provenance).
- **Alternatives**: the stretches without calls of the free bird recordings, screened by BirdNET (no extra key, but
  only quiet forest from birders' recordings, no wind or rain: rejected as the main source, kept as fallback if no
  Freesound key is available); ESC-50 (mostly CC BY-NC: rejected).
- **Metric**: false-detection rate = share of non-bird windows whose top class (float, and int8) is a species; and the
  same counting only those above the device's confidence threshold (50 %, `examples/birds/project.yaml`).

## R5. Statistics (spec FR-004)

- Accuracy, macro-F1, per-class recall and precision, the confusion matrix, counts of windows and recordings, for
  float and device int8, overall and for the unseen subset.
- **Intervals**: 95 % percentile bootstrap over recordings (windows of one recording resampled together), 2,000
  resamples, seed 0, so the output is deterministic.
- The float − int8 difference in macro-F1 is bootstrapped the same way (SC-003).

## R6. One command, reproducible (spec FR-006, SC-001)

- **Decision**: `python tools/measure_bird_accuracy.py [OUT_DIR]` (standard library + subprocess, like `python -m wren`):
  prepare with the checkpoint's class order and zero validation/test ratios (torch env) → `windows_ref` (torch env) → build `libfe`/`libtail` from
  `audiomoth/firmware/bird` with the shipped headers (cc, as the package stage does) → `device_eval` with the shipped
  `bird_backbone_int8.tflite` (tf env) → statistics (torch env, numpy) → `accuracy.json` and `accuracy.md`.
- The committed reference is `examples/birds/test/accuracy.json`; `tests/test_bird_accuracy.sh` reruns the measurement
  and compares `accuracy.json` byte for byte in the Docker image (CI), and prints the difference natively.
- **Determinism**: prepare is deterministic; windows_ref runs in float64 rounded once; device_eval uses TFLite's
  reference kernels; the bootstrap has a fixed seed; JSON written with sorted keys and fixed rounding.

## R7. Analogue-chain rows of the old §9

The simulated analogue chain (70/60/50 dB SPL) came from WrenNet scripts that are not in this repository. **Decision**:
not re-measured; the new §9 says they were removed and why (spec Assumptions).

## R8. Keys

The test bundle needs two maintainer keys, used once: `XC_API_KEY` (xeno-canto, as for the calibration bundle) and
`FREESOUND_API_KEY` (token). Both are in `~/.wren-keys` (outside the repository, never committed) and were checked on
2026-10-23: xeno-canto answers (1 CC0 recording of *Emberiza cia*), Freesound finds 10,519 CC0 sounds for "wind".
