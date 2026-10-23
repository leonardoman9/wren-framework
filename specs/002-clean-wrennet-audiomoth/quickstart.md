# Quickstart: validating the clean-up

Prerequisites: a working native install (`python -m wren setup --tflm`) or the Docker image; the repository on branch
`002-clean-wrennet-audiomoth`.

1. **Layout and stale names** (SC-003, SC-005):
   `.venv-torch/bin/python -m pytest -q tests/test_layout.py`
   → passes: `wrennet/` holds only the package, the checkpoint and the README; no `WRENNET_ROOT`,
   `birds_distillation_edge` or `best_distillation_model.pt` outside the `specs/` feature folders; none of the
   removed audiomoth files is tracked.

2. **Unit checks** (as CI):
   `.venv-torch/bin/python tests/test_training.py && .venv-torch/bin/python -m pytest -q tests/test_setup.py tests/test_examples.py tests/test_stage_keys.py tests/test_layout.py`
   → all pass.

3. **Bird reproduction** (SC-001):
   `tests/test_birds_reproduction.sh`
   → `identical` for the 7 package files and `PASS`; in Docker
   (`docker run --rm -v "$PWD/runs:/wren/runs" --entrypoint sh wren tests/test_birds_reproduction.sh`) also
   `identical firmware binaries`. The firmware stage inside it runs every firmware test.

4. **Synthetic end to end** (training path, `init: wrennet` from `wrennet/checkpoints/birds.pt`):
   `.venv-torch/bin/python tests/make_synthetic_dataset.py runs/synth_data && python -m wren run tests/synthetic/project.yaml`
   → every stage passes.

5. **A firmware script by hand** (US3):
   `.venv-torch/bin/python audiomoth/firmware/bird/gen_bird_tail.py wrennet/checkpoints/birds.pt runs/birds/export/reference.npz /tmp/tail`
   → writes the tail weights without any model-code path argument.

6. **CI** (SC-002): the docker, native-linux and native-macos jobs pass on the pull request.
