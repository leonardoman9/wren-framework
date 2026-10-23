#!/bin/sh
# The bird model's accuracy (specs/006-bird-accuracy-remeasure): re-measures it on the committed test bundle and compares
# the result with examples/birds/test/accuracy.json, the figures of audiomoth/docs/VERIFICATION.md §9. In the Docker image
# (WREN_PATH=docker) the file must be identical; natively the figures are printed and must agree within 0.02, window
# counts within 2: decoding and filtering differ by CPU in their last bits, which can move a call window across the
# selection threshold (measured: one window of 638 between macOS arm64 and Docker; constitution V).
# Usage: tests/test_bird_accuracy.sh
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${WREN_PYTHON:-.venv-torch/bin/python}
OUT=runs/bird-accuracy
"$PY" tools/measure_bird_accuracy.py "$OUT" > /dev/null
REF=examples/birds/test/accuracy.json
if [ "$WREN_PATH" = docker ]; then
  if cmp -s "$OUT/accuracy.json" "$REF"; then echo "PASS: accuracy.json identical"; else
    echo "FAIL: accuracy.json differs from $REF"; diff "$REF" "$OUT/accuracy.json" | head -40; exit 1; fi
else
  "$PY" - "$REF" "$OUT/accuracy.json" <<'PYEOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
diffs = []
def walk(x, y, path):
    if isinstance(x, dict):
        for k in set(x) | set(y): walk(x.get(k), y.get(k), f"{path}.{k}")
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (p, q) in enumerate(zip(x, y)): walk(p, q, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        limit = 2 if ("windows" in path or "confusion" in path or "recordings" in path) else 0.02
        if abs(x - y) > limit: diffs.append((path, x, y))
    elif x != y and "sha256" not in path: diffs.append((path, x, y))
walk(a, b, "")
for d in diffs[:20]: print("DIFFERENT", *d)
print("native install: figures within 0.02 (counts within 2) of the committed ones" if not diffs else f"FAIL: {len(diffs)} figures differ")
sys.exit(1 if diffs else 0)
PYEOF
fi
