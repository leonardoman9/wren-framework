# Contract: building the test bundle (maintainer, once)

`XC_API_KEY=… FREESOUND_API_KEY=… .venv-torch/bin/python tools/make_test_bundle.py OUT_DIR [--max-seconds 120]
[--non-bird 40]`

1. xeno-canto API v3: every recording of the 8 species with licence CC0 / CC BY / CC BY-SA (no NC, ND), quality A–C.
2. Per recording: decode, mono, 32 kHz; remove each calibration clip of that recording (`examples/birds/calibration/
   ATTRIBUTION.csv`, `start_s` .. `start_s + 3`) with 1.5 s on each side; keep stretches ≥ 3 s, at most `--max-seconds`
   in total, in recording order.
3. Freesound API v2: CC0 sounds for the queries `wind`, `rain`, `stream`, `river`, `insects`, `traffic`, `footsteps`,
   `forest ambience`, field recordings ≥ 10 s; first 30 s; BirdNET (feature 003, `third_party/birdnet`) on 3 s windows:
   drop the sound if any scientific-name label is above 0.1; stop at `--non-bird` sounds, balanced over the queries.
4. Write data-model.md's layout, `ATTRIBUTION.csv` (with `uploaded` from xeno-canto, Freesound's `created`),
   `SHA256SUMS`, `LICENSE.md`, `README.md`; print counts per species and the unseen recordings (uploaded after
   2025-06-11).

The committed bundle is the reference afterwards; the tool need not reproduce it byte for byte (as the calibration
bundle's tool).
