# Calibration bundle of the bird model

520 clips of 3 s (mono, 32 kHz, 16-bit FLAC) from freely licensed xeno-canto recordings (API v3): 146 from the model's 8 species, 374 from other European species, which complete it because only about 1 % of xeno-canto is freely licensed. Up to 2 windows per recording, the loudest non-overlapping ones (so that they contain calls), quality A/B/C, seed 0.

The pipeline uses it for the int8 calibration and evaluation (`calibration.audio`) and for the device test windows
(`test.audio`) of `examples/birds/project.yaml`. Changing any clip changes the int8 model and the test vector.
Built by `tools/make_calibration_bundle.py`; checked by `SHA256SUMS` and `tests/test_examples.py`.
