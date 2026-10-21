"""Checks the checkpoint and writes the reference data the firmware generators need.
Usage: python -m wren.torchside.reference CHECKPOINT.pt AUDIO OUT_DIR

- info.json: class count and learned front-end parameters (breakpoint, transition width, frequency range);
- reference.npz: one 3 s window of AUDIO (wav), its standardised spectrogram (spec), the model's logits, the
  learned filterbank: the input of firmware/bird/gen_bird_tail.py;
- the split check: front-end + core (what the device computes) must give the same logits as the whole model, and the
  core exported to ONNX the same as PyTorch.
(From WrenNet's tools/bird_model_export_check.py main(), with the general loader.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile
import torch
import torchaudio

from wren.torchside.model import CLIP_S, SR, Core, checkpoint_info, frontend, load_model

ckpt, audio, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
info = checkpoint_info(ckpt)
m = load_model(ckpt)
data, sr = soundfile.read(str(audio), dtype="float32", always_2d=True)
wav = torch.from_numpy(data.T.copy())
wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, SR)
n = int(SR * CLIP_S)
wav = torch.nn.functional.pad(wav, (0, max(0, n - wav.shape[1])))[:, :n]
with torch.no_grad():
    full = m(wav)
    feats, fb = frontend(m, wav)
    core = Core(m).eval()
    split = core(feats)
split_diff = float((full - split).abs().max())
onnx_path = out / "core.onnx"
torch.onnx.export(core, feats, str(onnx_path), opset_version=17, input_names=["spec"], output_names=["logits"], dynamo=False)
onnx.checker.check_model(onnx.load(str(onnx_path)))
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
onnx_diff = float(np.abs(sess.run(None, {"spec": feats.numpy()})[0] - split.numpy()).max())
np.savez(out / "reference.npz", wav=wav.numpy(), spec=feats.numpy(), logits=full.detach().numpy(), filterbank=fb.numpy())
info.update(split_max_diff=split_diff, onnx_max_diff=onnx_diff, audio=str(audio))
(out / "info.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info))
if split_diff > 1e-4 or onnx_diff > 1e-3:
    sys.exit(f"the split model does not reproduce the checkpoint (split {split_diff}, onnx {onnx_diff})")
