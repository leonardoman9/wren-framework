# Contract: `python -m wren setup --birdnet`

- Standard library only (as the rest of `wren/setup.py`). Combinable with `--tflm`; idempotent.
- Steps: if `third_party/birdnet/v2.4/.wren-birdnet` matches the pinned hashes → "BirdNET: ready", nothing downloaded.
  Otherwise download `https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip?download=1` to a temporary
  file, check sha256 `31377e128d86fe7b65fa91b206b8804bad1cd934e624bbe6e7c1788170c95e57`, extract `audio-model.tflite`
  and `labels/en_us.txt`, check their sha256 (data-model.md), write the marker last, delete the zip.
- Failure (network, checksum): plain message, nothing half-installed left behind (no marker), exit non-zero.
- Prints the licence and citation once: "BirdNET v2.4 (Kahl et al. 2021) is licensed CC BY-NC 4.0: non-commercial use.
  Models trained with it as a teacher may fall under the same condition."
- `python -m wren setup --check` adds the row "Learning from BirdNET (optional)": ready / "run python -m wren setup
  --birdnet". It is not required for "Ready".
- Docker: `RUN python -m wren setup --birdnet` in its own layer after `--tflm`.
