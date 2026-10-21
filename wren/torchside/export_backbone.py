# Copied unchanged from WrenNet tools/bird_export_backbone.py (not published in the WrenNet repository). Run through
# wren/torchside/run.py, which gives it the general model loader (any number of classes).
"""Export the convolutional backbone of the bird model to ONNX ([1,64,301] -> [1,32,301]),
optionally clamping the output to [0, CLIP] (outlier activations at the clip start would
otherwise dictate the int8 range).
Usage: python tools/bird_export_backbone.py CHECKPOINT.pt OUT.onnx [CLIP|none] [2d|1d]   (nn.Hardswish is rewritten for TFLM)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bird_model_export_check import load_model  # noqa: E402


def _hswish(self, x):
    """x * relu6(x + 3) / 6: the same function as nn.Hardswish, built from Add/Relu6/Mul, which
    TFLite Micro implements (nn.Hardswish exports as HardSigmoid -> RELU_0_TO_1, which it does not).
    Patched on the class because some Hardswish instances are not registered sub-modules."""
    return x * torch.nn.functional.relu6(x + 3.0) * (1.0 / 6.0)


def to_2d(module):
    """Rewrites the 1-D convolutional stack as an equivalent 4-D [N, C, 1, T] network (same weights).
    TFLite only has 2-D convolutions, so the converter wraps every 1-D op in Reshapes (74 of them
    in this model); expressing the network in 4-D from the start avoids them."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d):
            new = nn.Conv2d(child.in_channels, child.out_channels, (1, child.kernel_size[0]),
                            stride=(1, child.stride[0]), padding=(0, child.padding[0]),
                            dilation=(1, child.dilation[0]), groups=child.groups, bias=child.bias is not None)
            new.weight.data = child.weight.data.unsqueeze(2).clone()
            if child.bias is not None:
                new.bias.data = child.bias.data.clone()
            setattr(module, name, new)
        elif isinstance(child, nn.BatchNorm1d):
            new = nn.BatchNorm2d(child.num_features, eps=child.eps, momentum=child.momentum)
            new.load_state_dict(child.state_dict())
            setattr(module, name, new)
        elif isinstance(child, nn.ConstantPad1d):
            setattr(module, name, nn.ConstantPad2d((child.padding[0], child.padding[1], 0, 0), child.value))
        elif isinstance(child, nn.Dropout1d):
            setattr(module, name, nn.Identity())
        else:
            to_2d(child)


class Backbone(nn.Module):
    def __init__(self, phi, clip):
        super().__init__()
        self.phi, self.clip = phi, clip

    def forward(self, x):
        y = self.phi(x)
        return y if self.clip is None else torch.clamp(y, 0.0, self.clip)


m = load_model(sys.argv[1])
x_check = torch.randn(1, 64, 301)
with torch.no_grad():
    before = m.phi(x_check)
nn.Hardswish.forward = _hswish
with torch.no_grad():
    print("hardswish replacement max |diff| =", float((m.phi(x_check) - before).abs().max()))
clip = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "none" else None
mode = sys.argv[4] if len(sys.argv) > 4 else "2d"
if mode == "2d":
    import torch.nn.functional as F
    reference = m.phi.eval()
    with torch.no_grad():
        ref_out = reference(x_check)
    to_2d(m.phi)
    F.adaptive_avg_pool1d = lambda x, size: F.adaptive_avg_pool2d(x, (1, 1))  # squeeze-excitation pooling over time
    m.phi.eval()
    with torch.no_grad():
        out2d = m.phi(x_check.unsqueeze(2))
    print("4-D rewrite max |diff| =", float((out2d.squeeze(2) - ref_out).abs().max()))
    example = torch.zeros(1, 64, 1, 301)
else:
    example = torch.zeros(1, 64, 301)
torch.onnx.export(Backbone(m.phi.eval(), clip).eval(), example, sys.argv[2], opset_version=13,
                  input_names=["spec"], output_names=["feat"], dynamo=False)
print("exported", sys.argv[2], "clip", clip)
