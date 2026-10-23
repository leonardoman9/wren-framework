# Quickstart: validating the re-measured accuracy

Prerequisites: `python -m wren setup --tflm`; for step 1 only, `python -m wren setup --birdnet` and the two API keys.

1. **Bundle (maintainer, once)**: contracts/test-bundle.md → `examples/birds/test/` with attribution and checksums;
   `pytest tests/test_examples.py` passes (licences, attribution rows, no BioDCASE).
2. **Statistics on a fixture**: `.venv-torch/bin/python -m pytest -q tests/test_measure.py` → bootstrap per recording,
   false detections, "not meaningful" below 5 recordings.
3. **Prepare unchanged for training**: `.venv-torch/bin/python tests/test_training.py` and the synthetic project pass.
4. **Measure**: `python tools/measure_bird_accuracy.py` → `runs/bird-accuracy/accuracy.{json,md}`; run twice: identical.
5. **Reproduce**: `tests/test_bird_accuracy.sh` → natively within 0.002 of the committed `accuracy.json`; in Docker
   identical (CI).
6. **Documents**: §9 of `audiomoth/docs/VERIFICATION.md` and the package README show `accuracy.md`'s figures, method
   and limits; the old figures are gone from them.
