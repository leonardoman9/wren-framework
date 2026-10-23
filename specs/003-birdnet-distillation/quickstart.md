# Quickstart: validating "learn from BirdNET"

Prerequisites: `python -m wren setup --tflm` done; branch `003-birdnet-distillation`.

1. **Install BirdNET**: `python -m wren setup --birdnet` → downloads ~77 MB once, prints the licence; a second run
   prints "ready" and downloads nothing. `python -m wren setup --check` shows the BirdNET row as ready.
2. **Unit checks**: `.venv-torch/bin/python -m pytest -q tests/test_birdnet.py` → target, loss, species matching and
   refusals pass (no download needed).
3. **Refusal (US2, SC-001)**: `python -m wren run tests/synthetic/project_birdnet.yaml` (the synthetic sounds with
   `birdnet: true`) → stops within seconds, naming `upsweep`, `downsweep`, `trill`.
4. **Bird project end to end (US1, SC-002)**:
   `.venv-torch/bin/python tests/make_bird_dataset.py runs/bird_data && python -m wren run tests/birds_birdnet/project.yaml`
   → every stage passes; the report has the BirdNET line, licence and agreement per species.
5. **Cache (SC-004)**: run step 4 again with `--stages train --force` → "asking BirdNET" is skipped (its results are
   reused), training runs.
5b. **Without BirdNET (SC-006)**: move `third_party/birdnet` aside, run step 4's project with `--stages train,evaluate
   --force` → training repeats with the stored results; move it back.
6. **Comparison (SC-003)**: `python -m wren run tests/birds_birdnet/project_off.yaml --stages prepare,train,evaluate`
   → compare test macro-F1 with step 4; record both.
7. **Option off unchanged (SC-005)**: `tests/test_training.py`, the synthetic project and the bird reproduction pass as
   on `main`; the synthetic project's `train/train.json` (best epoch, metrics) and `train/history.csv` are identical to
   those of a run on `main` on the same computer.
