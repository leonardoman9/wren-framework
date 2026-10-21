# wren-framework with both Python environments, TFLM for the firmware and the firmware tests' tools.
#   docker build -f docker/Dockerfile -t wren .                         (from the repository root, submodules checked out)
#   docker run --rm -v "$PWD/runs:/wren/runs" -v /path/to/audio:/audio wren run examples/.../project.yaml
FROM python:3.11.9-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential clang git curl unzip make mtools qemu-system-arm libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wren
COPY envs/ envs/
RUN python -m venv .venv-torch && .venv-torch/bin/pip install --no-cache-dir -r envs/requirements-torch.txt \
 && python -m venv .venv-tf && .venv-tf/bin/pip install --no-cache-dir -r envs/requirements-tf.txt

COPY . .
# TFLM at the commit the firmware was verified with, built for the Cortex-M4F and for the host (simulator tests)
RUN sh third_party/audiomoth-fw/scripts/setup_tflm.sh && HOST=1 sh third_party/audiomoth-fw/scripts/build_tflm.sh

ENTRYPOINT [".venv-torch/bin/python", "-m", "wren"]
