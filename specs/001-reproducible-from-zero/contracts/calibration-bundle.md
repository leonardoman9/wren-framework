# Contract: calibration bundle

The audio the bird model is quantised and tested with: 3 s clips from xeno-canto recordings (the model's 8
species first, then other European species), licensed CC0, CC-BY or CC-BY-SA only, committed to the repository with per-clip attribution. No
BioDCASE audio is allowed here or anywhere else in the project (FR-014).

## Layout

```text
examples/birds/calibration/
├── audio/<id>.flac   # ≥ 400 clips: mono, PCM_16, 32 kHz, exactly 96,000 samples (3 s)
├── SHA256SUMS        # sha256sum format, one line per audio/<id>.flac
├── ATTRIBUTION.csv   # clip,xc_id,species,recordist,licence,licence_url,source_url,start_s
├── LICENSE.md        # each clip keeps its CC licence; clips are modified (cut, resampled, re-encoded)
└── README.md         # source (xeno-canto API v3), selection rule, how the clips were cut
```

`<id>` is `XC<xc_id>_<start-seconds as 4 digits>`, for example `XC106810_0045`. Ids are unique and
contain only ASCII characters.

## `project.yaml` (existing keys only)

```yaml
calibration:
  audio: [calibration/audio]   # relative to project.yaml
  count: 300
  eval: 100
test:
  audio: [calibration/audio]
  windows: 120
```

Selection is the existing behaviour: sorted file list, then a fixed-seed shuffle (seed 0 for calibration in
`make_specs.py`, seed 7 for test windows in `e2e_make_ref.py`). The reference clip is the first file in
sorted order. Because the folder is fixed, the selection is fixed.

## Required code changes

| File | Change |
|---|---|
| `wren/stages/export.py` `audio_files` | Glob `*.flac` in addition to `*.wav` and `*.mp3` |
| `wren/torchside/make_specs.py` | Same glob change. The header note "copied unchanged" becomes "copied from WrenNet; +flac" |
| `audiomoth/firmware/bird/e2e_make_ref.py` | Same glob change |

## Builder (maintainer only, run once)

```text
XC_API_KEY=… .venv-torch/bin/python tools/make_calibration_bundle.py OUT_DIR [--clips 400] [--test 120] [--per-file 2] [--seed 0]
```

- Species order:
  1. the 8 species of `examples/birds/project.yaml` (not `non-bird`);
  2. then the other 62 species of the project's 70-species list, written in the builder as `FILL_SPECIES`:
     `Accipiter gentilis`, `Aegithalos caudatus`, …, `Upupa epops`.
- For each species and each free licence it queries
  `https://xeno-canto.org/api/3/recordings?query=gen:<Genus> sp:<species> lic:<pd|by|by-sa>&key=$XC_API_KEY&page=N`
  over all pages. `lic:pd` is CC0. The tags were verified on 2026-10-22: for *Bubo bubo* they return the same 10
  recordings as scanning every page.
- It keeps recordings whose `lic` URL is CC0 (`publicdomain/zero`), CC-BY (`licenses/by/`) or CC-BY-SA
  (`licenses/by-sa/`), with quality `q` A, B or C. Any `-nc` or `-nd` licence is rejected.
- It takes every kept recording of the 8 model species first (up to `--per-file` windows each). It then
  fills the remaining windows from the other species, round-robin across species, in xeno-canto ID order
  shuffled with `--seed`, until it has `--clips` (calibration and evaluation) plus `--test` windows. The
  selection does not need to be reproducible later; the committed bundle is the reference.
- It downloads only the selected `file` URLs, then decodes with soundfile (MP3 supported) and downmixes to
  mono. It resamples to 32 kHz with `torchaudio.functional.resample`, then picks up to `--per-file`
  non-overlapping 3 s windows with the highest RMS. Conversion to int16 uses `rint` and clipping (the same
  conversion as `e2e_make_ref.py`).
- It writes the FLAC files, `SHA256SUMS`, `ATTRIBUTION.csv`, `LICENSE.md` and `README.md`.
- It refuses to write a clip whose licence is not in the allowed set, and refuses any path containing
  "biodcase".
- The API key is read from the environment only. It is never written to any file, and users and CI never
  need it.

## Checks

- `tests/test_birds_reproduction.sh` runs `sha256sum -c SHA256SUMS` first and stops with the failing clip's
  name.
- `tests/test_examples.py` fails if any YAML under `examples/` or `tests/` has an absolute path or the string
  "biodcase" (case-insensitive), if `ATTRIBUTION.csv` and `audio/` do not match one to one, or if a licence is
  not CC0, CC-BY or CC-BY-SA.
