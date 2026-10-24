# Data Model: Re-measure the bird model's accuracy

## Test bundle (`examples/birds/test/`)

| Path | Content |
|------|---------|
| `audio/<Genus_species>/XC<id>/XC<id>_<k>.flac` | stretch k (≥ 3 s) of xeno-canto recording `id`, outside its calibration clips (± 1.5 s), ≤ 120 s per recording, 32 kHz mono 16-bit FLAC |
| `audio/non-bird/FS<id>/FS<id>.flac` | Freesound CC0 field recording, ≤ 30 s, no BirdNET bird species above 0.1 in any 3 s window |
| `ATTRIBUTION.csv` | per clip: `clip, source (xeno-canto/freesound), source_id, species (or non-bird), recordist, licence, licence_url, source_url, start_s, end_s, uploaded` |
| `SHA256SUMS` | every audio file |
| `LICENSE.md`, `README.md` | licences per clip; what the bundle is, how it was built, its limits |
| `accuracy.json` | the committed measurement (reference) |

Rules: licence CC0 / CC BY / CC BY-SA only (no NC, ND); no BioDCASE; folder names in the checkpoint's class order
are `Bubo_bubo`, `Certhia_familiaris`, `Apus_apus`, `Certhia_brachydactyla`, `Emberiza_cia`, `Lophophanes_cristatus`,
`Periparus_ater`, `Poecile_montanus`, `non-bird`.

## Measurement (`accuracy.json`)

```json
{"package_sha256": "<bird_backbone_int8.tflite>", "checkpoint_sha256": "…", "bundle_sha256": "<SHA256SUMS>",
 "confidence_threshold": 0.5,
 "classes": ["Bubo bubo", "…", "non-bird"], "unseen_after": "2025-06-11", "min_recordings": 5,
 "bootstrap": {"resamples": 2000, "seed": 0, "level": 0.95},
 "subsets": {
   "all":    {"float": {…}, "int8": {…}, "int8_minus_float_macro_f1": {"value": -0.01, "ci": [-0.03, 0.01]}},
   "unseen": {…same shape, or null if no recording…}},
 "per_class": {"<class>": {"recordings": 9, "windows": 41, "meaningful": true}}}
```

Per model version (`float`, `int8`): `accuracy` and `macro_f1` as `{"value", "ci": [lo, hi]}`, `recall` and `precision`
per class, `confusion` (rows = true class, model order), `false_detection` (share of non-bird windows whose top class
is a species) and `false_detection_confident` (same, top probability ≥ `confidence_threshold`, read from
`examples/birds/project.yaml` `device.confidence`, 50 % today). Numbers rounded to 4 decimals; keys
sorted.

`accuracy.md`: the same as tables in plain words, the source of §9.
