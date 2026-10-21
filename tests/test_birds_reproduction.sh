#!/bin/sh
# Regression test of the conversion: running examples/birds from the WrenNet checkpoint must give, byte for byte, the
# model package shipped in the firmware repository (firmware/bird/model), which was made and verified by hand.
# Any change to the conversion code or to the pinned environments must keep this passing.
# Usage: tests/test_birds_reproduction.sh   (needs the calibration audio listed in examples/birds/project.yaml)
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${WREN_PYTHON:-.venv-torch/bin/python}
"$PY" -m wren run examples/birds/project.yaml --stages export,package --force
A=runs/birds/package; B=third_party/audiomoth-fw/firmware/bird/model
fail=0
for pair in "model.tflite:bird_backbone_int8.tflite" generated/bird_frontend_tables.h generated/bird_tail_weights_i8.h \
            generated/tail_dims.h generated/model_config.h model_config.json test/pipe_vec.cc; do
  a=${pair%%:*}; b=${pair##*:}
  if cmp -s "$A/$a" "$B/$b"; then echo "identical  $a"; else echo "DIFFERENT  $a (vs $B/$b)"; fail=1; fi
done
[ $fail = 0 ] && echo "PASS: the pipeline reproduces the shipped bird package" || { echo "FAIL"; exit 1; }
