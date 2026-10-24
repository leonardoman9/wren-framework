"""BirdNET as a screen for non-bird test audio (tf environment; specs/006-bird-accuracy-remeasure, research R4).
Usage: python birdnet_screen.py BIRDNET_DIR IN_DIR OUT.json

IN_DIR holds <name>.npy files: mono int16 audio at 48 kHz. For each, BirdNET scores every 3 s window (3 s hop, the last
one padded) and OUT.json gets, per name, the highest confidence of any species label (any animal with a scientific
name, not BirdNET's 'Dog', 'Engine', 'Noise', ... classes) and that species."""
import json
import sys
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

WIN48 = 144000
NOT_SPECIES = {"Dog", "Engine", "Environmental", "Fireworks", "Gun", "Human non-vocal", "Human vocal", "Human whistle",
               "Noise", "Power tools", "Siren"}
birdnet_dir, in_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
names = [line.split("_", 1)[0] for line in (birdnet_dir / "labels" / "en_us.txt").read_text().splitlines()]
species = np.array([n not in NOT_SPECIES for n in names])

it = Interpreter(model_path=str(birdnet_dir / "audio-model.tflite"), num_threads=1)
it.allocate_tensors()
inp, outp = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
result = {}
for f in sorted(in_dir.glob("*.npy")):
    a = np.load(f).astype(np.float32) / 32768.0
    best, who = 0.0, None
    for s in range(0, max(1, len(a) - WIN48 + 1), WIN48):
        x = a[s: s + WIN48]
        if len(x) < WIN48: x = np.pad(x, (0, WIN48 - len(x)))
        it.set_tensor(inp, x[None]); it.invoke()
        p = 1.0 / (1.0 + np.exp(-it.get_tensor(outp)[0].astype(np.float64)))
        p[~species] = 0.0
        if p.max() > best: best, who = float(p.max()), names[int(p.argmax())]
    result[f.stem] = {"max_species_confidence": round(best, 6), "species": who}
out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({"screened": len(result)}))
