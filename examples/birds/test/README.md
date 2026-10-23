# Test bundle of the bird model

143 clips, 46 minutes, for measuring the shipped bird model's accuracy (`python tools/measure_bird_accuracy.py`, specs/006-bird-accuracy-remeasure). Recordings per class: Apus apus 4, Bubo bubo 9, Certhia brachydactyla 10, Certhia familiaris 6, Emberiza cia 4, Lophophanes cristatus 9, Periparus ater 15, Poecile montanus 4, non-bird 38.

- **Species**: every freely licensed xeno-canto recording of the model's 8 species (CC0, CC BY, CC BY-SA; quality A-C), without the stretches the calibration bundle took from it (each calibration clip +-1.5 s), at most 120 s per recording, in stretches of at least 3 s.
- **Non-bird**: CC0 Freesound field recordings (wind, rain, stream, river, insects, traffic, footsteps, forest ambience), the first 30 s, kept only if BirdNET v2.4 hears no species above 0.1 in any 3 s window.

Limits: the species clips come from the same recordings as the calibration clips (other times), and WrenNet was
trained on xeno-canto audio, so overlap with its training data cannot be excluded, except for the recordings
uploaded after 2025-06-11 (`uploaded` in `ATTRIBUTION.csv`), which the measurement reports separately.

Built by `tools/make_test_bundle.py`; checked by `SHA256SUMS` and `tests/test_examples.py`. No BioDCASE audio.
