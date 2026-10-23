"""BirdNET's opinion of the training windows (tf environment: ai_edge_litert; specs/003-birdnet-distillation).
Usage: python birdnet.py BIRDNET_DIR WORK_DIR INDICES_JSON OUT_DIR

WORK_DIR holds audio48.raw and grid.json (wren/torchside/birdnet_windows.py). INDICES_JSON lists BirdNET's output index
of each project species, in class order without the background. Writes OUT_DIR/teacher.npz: per window the source
file's sha256 and the start (32 kHz samples), the confidence (sigmoid of BirdNET's logit) of each project species, and
BirdNET's most confident label overall by scientific name, with its confidence."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

WIN48 = 144000
birdnet_dir, work, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[4])
indices = json.loads(Path(sys.argv[3]).read_text())
names = [line.split("_", 1)[0] for line in (birdnet_dir / "labels" / "en_us.txt").read_text().splitlines()]
g = json.loads((work / "grid.json").read_text())
audio = np.memmap(work / "audio48.raw", dtype=np.int16, mode="r")

it = Interpreter(model_path=str(birdnet_dir / "audio-model.tflite"), num_threads=1)
it.allocate_tensors()
inp, outp = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
t0 = time.time()
sha, start, conf, top_name, top_conf = [], [], [], [], []
for fi, s in g["grid"]:
    f = g["files"][fi]
    a = f["offset"] + s * 3 // 2  # 32 -> 48 kHz
    x = audio[a: min(a + WIN48, f["offset"] + f["length"])].astype(np.float32) / 32768.0
    if len(x) < WIN48: x = np.pad(x, (0, WIN48 - len(x)))
    it.set_tensor(inp, x[None])
    it.invoke()
    p = 1.0 / (1.0 + np.exp(-it.get_tensor(outp)[0].astype(np.float64)))
    sha.append(f["sha256"]); start.append(s); conf.append(p[indices])
    top = int(p.argmax()); top_name.append(names[top]); top_conf.append(p[top])
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "teacher.npz", file_sha256=np.array(sha), start=np.array(start, dtype=np.int64),
         confidence=np.array(conf, dtype=np.float32).reshape(len(sha), len(indices)),
         top_name=np.array(top_name), top_confidence=np.array(top_conf, dtype=np.float32))
print(json.dumps({"windows": len(sha), "seconds": round(time.time() - t0, 1)}))
