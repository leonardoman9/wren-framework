# Contract: the measurement command

`python tools/measure_bird_accuracy.py [OUT_DIR]` (default `runs/bird-accuracy/`), any Python 3.11 (it starts the
environments' Pythons itself, like `python -m wren run`).

Every torch and tf step runs with `wren.runner.PINNED` in its environment (fixed CPU kernels and one thread, as the
export and package stages), so the output is the same bytes on every x86 computer in Docker (FR-006).

Steps (logs in `OUT_DIR/logs/`):
1. Check `examples/birds/test/SHA256SUMS`; stop with a plain message if a file differs.
2. `wren.torchside.prepare` with classes in the checkpoint's order (folder names, `non-bird` last), sources
   `{"train": examples/birds/test/audio}`, `split: {validation: 0, test: 0, seed: 0}`, `group: subfolder`, default
   windows → every recording's call windows and non-bird windows in the `train` split.
3. `wren.torchside.windows_ref OUT/prepare train wrennet/checkpoints/birds.pt OUT/ref.npz 1000000 0` (all windows).
4. `cc -O2 -shared -fPIC` `bird_frontend.c` and `bird_tail.c` with `-I audiomoth/firmware/bird/model/generated`.
5. `wren.tfside.device_eval OUT/ref.npz audiomoth/firmware/bird/model/bird_backbone_int8.tflite libfe libtail 9
   OUT/device_eval.json OUT/windows.npz` (per-window labels, float and int8 logits).
6. `wren.torchside.accuracy` → `OUT/accuracy.json`, `OUT/accuracy.md` (data-model.md); each window's recording from
   `ref.npz` `names` (`XC<id>_<k>.flac@<s>s` / `FS<id>.flac@<s>s`: the `XC<id>` / `FS<id>` prefix), `uploaded` from
   `ATTRIBUTION.csv`, the confidence threshold from `examples/birds/project.yaml` (`device.confidence`).

Exit 0 and a one-paragraph summary on success; plain message and exit 1 on failure.

`tests/test_bird_accuracy.sh`: runs the command and compares `OUT/accuracy.json` with `examples/birds/test/accuracy.json`
— must be identical in the Docker image (`WREN_PATH=docker`); natively prints the differences and passes when every
figure is within 0.002.
