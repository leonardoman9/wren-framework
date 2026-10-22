# Contract: Docker image

## Build (users and CI)

```sh
git clone https://github.com/<org>/wren-framework && cd wren-framework
docker build --platform linux/amd64 -f docker/Dockerfile -t wren .
```

No image is published. Every build starts from the same pinned inputs:

| Input | Pinned by |
|---|---|
| Base image | `python:3.11.9-slim-bookworm@sha256:<digest>`, `--platform=linux/amd64` |
| System packages | apt sources at `snapshot.debian.org/archive/debian/<timestamp>`: build-essential, clang and libclang-rt-14-dev (firmware simulator with sanitizers), git, wget, curl, unzip, make, mtools, qemu-system-arm, libsndfile1 |
| Python packages | `envs/lock/{torch,tf}-linux-x86_64.txt` with `--require-hashes` |
| TFLM | commit e0021ca + every patch in `audiomoth/firmware/patches/` (`audiomoth/scripts/setup_tflm.sh`); its layer copies only those patches and the scripts |
| ARM GCC | 14.3.Rel1 x86_64, MD5-checked by TFLM's download script |
| Repository | the build context (no submodules; `.dockerignore` excludes `runs/`, `.venv-*`, a local `tflite-micro` and build output inside `audiomoth/`) |

## Run

```sh
docker run --rm --platform linux/amd64 -v "$PWD/runs:/wren/runs" -v /path/to/audio:/audio \
  wren run /wren/examples/birds/project.yaml
```

- The entrypoint is `.venv-torch/bin/python -m wren`, so every `wren` subcommand works as an argument.
- Outputs are written to `/wren/runs` (mount it to keep them).
- The image sets `WREN_PATH=docker` and `WREN_IMAGE_ID` (a build arg) for the report's provenance block.

## Guarantees

- For the same commit, every build of this Dockerfile on any host gives byte-identical deterministic outputs
  (export, package and firmware): SC-004 and clarification 2.
- `tests/test_birds_reproduction.sh` passes inside the image, including the firmware `.bin` sha256.
