#!/bin/sh
# Regression test of the conversion: running examples/birds (the WrenNet checkpoint and the committed calibration bundle)
# must give, byte for byte, the bird package in audiomoth/firmware/bird/model and, in the Docker image (WREN_PATH=docker),
# the firmware binaries recorded in tests/expected/birds_firmware.sha256. Natively the binaries' hashes are only printed:
# byte-identity is guaranteed on the Docker reference path (constitution V); a native install is held to the parity
# checks, which the pipeline runs anyway. Any change to the conversion code or to the locked environments must keep this
# passing in Docker.
# Usage: tests/test_birds_reproduction.sh
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${WREN_PYTHON:-.venv-torch/bin/python}
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

(cd examples/birds/calibration && sha -c --quiet SHA256SUMS) \
  || { echo "FAIL: the calibration bundle differs from examples/birds/calibration/SHA256SUMS"; exit 1; }

"$PY" -m wren run examples/birds/project.yaml --stages export,package,firmware --force
A=runs/birds/package; B=audiomoth/firmware/bird/model
fail=0
for pair in "model.tflite:bird_backbone_int8.tflite" generated/bird_frontend_tables.h generated/bird_tail_weights_i8.h \
            generated/tail_dims.h generated/model_config.h model_config.json test/pipe_vec.cc; do
  a=${pair%%:*}; b=${pair##*:}
  if cmp -s "$A/$a" "$B/$b"; then echo "identical  $a"; else echo "DIFFERENT  $a (vs $B/$b)"; fail=1; fi
done

F=runs/birds/firmware; E=tests/expected/birds_firmware.sha256
if [ "$WREN_PATH" = docker ]; then
  if (cd "$F" && sha -c "$ROOT/$E"); then echo "identical  firmware binaries"; else echo "DIFFERENT  firmware binaries (vs $E)"; fail=1; fi
else
  echo "native install: firmware hashes for information (compared only in Docker):"; (cd "$F" && sha birds.bin birds_diag.bin)
fi
[ $fail = 0 ] && echo "PASS: the pipeline reproduces the shipped bird package" || { echo "FAIL"; exit 1; }
