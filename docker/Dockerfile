# wren-framework, the byte-reproducible reference path (contracts/docker-image.md). No image is published: build it.
#   docker build --platform linux/amd64 -f docker/Dockerfile -t wren .             (from the repository root)
#   docker run --rm --platform linux/amd64 -v "$PWD/runs:/wren/runs" -v /path/to/audio:/audio wren run PROJECT.yaml
# Every input is pinned, so builds made at different times give the same environment: the base image by digest
# (linux/amd64 on any host: the float conversion and the compiler differ by CPU architecture), Debian packages from
# a dated snapshot, Python packages from the hash-checked locks, TFLM and its ARM compiler by commit and checksum.
FROM --platform=linux/amd64 python:3.11.9-slim-bookworm@sha256:2856e6af199e8128161abd320575eb9b341f3b76f017b5d0c9cd364f60d8a050

ARG DEBIAN_SNAPSHOT=20261015T000000Z
RUN rm -f /etc/apt/sources.list.d/debian.sources \
 && printf '%s\n' \
      "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT} bookworm main" \
      "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT} bookworm-updates main" \
      "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT} bookworm-security main" \
      > /etc/apt/sources.list \
 && apt-get update && apt-get install -y --no-install-recommends \
        build-essential clang libclang-rt-14-dev git wget curl unzip make mtools qemu-system-arm libsndfile1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /wren
# the same setup command as a native install (FR-006), in layers that change rarely
COPY envs/ envs/
COPY wren/__init__.py wren/__main__.py wren/cli.py wren/setup.py wren/
RUN python -m wren setup || { cat runs/_setup/logs/setup.log; exit 1; }
# only what TFLM needs, so that changes elsewhere in audiomoth/ (code, model package) do not rebuild it
COPY audiomoth/scripts/ audiomoth/scripts/
COPY audiomoth/firmware/patches/ audiomoth/firmware/patches/
RUN python -m wren setup --tflm || { tail -50 runs/_setup/logs/setup.log; exit 1; }

COPY . .
ARG WREN_IMAGE_ID=unset
ENV WREN_PATH=docker WREN_IMAGE_ID=${WREN_IMAGE_ID}
ENTRYPOINT [".venv-torch/bin/python", "-m", "wren"]
