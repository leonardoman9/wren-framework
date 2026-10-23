"""Training audio for BirdNET (torch environment; specs/003-birdnet-distillation, research R4).
Usage: python -m wren.torchside.birdnet_windows PREP_DIR HOP_SECONDS HASHES_JSON OUT_DIR

Every training file of prepare (audio_train.raw, 32 kHz) is resampled once to BirdNET's 48 kHz and stored as int16 in
OUT_DIR/audio48.raw; grid.json lists its 3 s windows on prepare's grid (start 0, h, 2h, ... up to the file's end), with
each file identified by the sha256 of its source audio (HASHES_JSON: path -> sha256), not by its index, so BirdNET's
results can be reused wherever the same recordings are trained on."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

WIN, SR, SR_BIRDNET = 96000, 32000, 48000
prep, hop, hashes, out = Path(sys.argv[1]), float(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
files = json.loads((prep / "files_train.json").read_text())
audio = np.memmap(prep / "audio_train.raw", dtype=np.int16, mode="r", shape=(sum(f["length"] for f in files),))
h = int(round(hop * SR))

stored, grid, offset = [], [], 0
with open(out / "audio48.raw", "wb") as raw:
    for fi, f in enumerate(files):
        pcm = torch.from_numpy(audio[f["offset"]: f["offset"] + f["length"]].astype(np.float32) / 32768.0)
        x = torchaudio.functional.resample(pcm, SR, SR_BIRDNET)
        x = np.clip(np.rint(x.numpy() * 32768.0), -32768, 32767).astype(np.int16)
        raw.write(x.tobytes())
        stored.append({"sha256": hashes[f["path"]], "offset": offset, "length": len(x)})
        offset += len(x)
        grid += [[fi, s] for s in (list(range(0, f["length"] - WIN + 1, h)) or [0])]
(out / "grid.json").write_text(json.dumps({"files": stored, "grid": grid}))
print(json.dumps({"files": len(files), "windows": len(grid), "seconds_48k": round(offset / SR_BIRDNET, 1)}))
