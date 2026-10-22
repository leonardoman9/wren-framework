"""WrenNet model loading and the parts of it the conversion needs (runs in the torch environment).

The architecture is fixed (WrenNet: 32 kHz, 3 s, STFT 1024/320, 64 learned log-linear filters, PhiNet/MatchboxNet
backbone, GRU 64, attention); the number of classes and the learned filterbank parameters come from the checkpoint.
The architecture code is WrenNet's own (wrennet/, or WRENNET_ROOT).
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

WRENNET_ROOT = Path(os.environ.get("WRENNET_ROOT") or Path(__file__).resolve().parents[2] / "wrennet")
sys.path.insert(0, str(WRENNET_ROOT))
from birds_distillation_edge.models import Improved_Phi_GRU_ATT  # noqa: E402

SR, CLIP_S = 32000, 3.0


def checkpoint_info(path):
    """Number of classes and learned front-end parameters, read from the state dict."""
    sd = torch.load(path, map_location="cpu")
    return {
        "classes": int(sd["fc.weight"].shape[0]),
        "breakpoint": float(sd["combined_log_linear_spec.breakpoint"]),
        "transition_width": float(sd["combined_log_linear_spec.transition_width"]),
        "f_min": float(sd["combined_log_linear_spec.f_min"]),
        "f_max": float(sd["combined_log_linear_spec.f_max"]),
    }


F_MIN, F_MAX = 150.0, 16000.0  # the filterbank's range (buffers, not learned), as in the bird model


def build_model(classes, breakpoint=4000.0, transition_width=100.0):
    """WrenNet with `classes` outputs: the architecture and hyper-parameters of the bird model (hidden size 64, 32
    backbone channels, 64 filters). The `matchbox` keys below are not all read by WrenNet (MatchboxNetSkip uses its own
    defaults: dropout 0.3 in its blocks, 0.05 in the PhiNet blocks); kept as in the original export script."""
    m = Improved_Phi_GRU_ATT(
        num_classes=classes, spectrogram_type="combined_log_linear", sample_rate=SR, n_mel_bins=64,
        n_linear_filters=64, hidden_dim=64, n_fft=1024, hop_length=320,
        matchbox=dict(base_filters=32, num_layers=3, kernel_size=3, dropout=0.1),
        breakpoint=breakpoint, transition_width=transition_width, f_min=F_MIN, f_max=F_MAX)
    fb = m.combined_log_linear_spec._get_current_filter_bank(device=torch.device("cpu"))
    if not torch.isfinite(fb).all(): raise ValueError("the filterbank of a new model is not finite")
    return m


def load_model(path):
    """The trained network, in eval mode; the class count is the checkpoint's."""
    m = build_model(checkpoint_info(path)["classes"])
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return m.eval()


def power_to_db(power):
    """10 log10(power) with the 80 dB floor below EACH CLIP's maximum, as the C front-end does. (torchaudio's
    AmplitudeToDB, used inside the WrenNet model, takes one floor for the whole batch: batched results then differ
    from one clip at a time and from the device. For one clip both are the same.)"""
    db = 10.0 * torch.log10(torch.clamp(power, min=1e-10))
    return torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - 80.0)


def frontend(m, wav):
    """The part of the model the device computes in C: STFT, learned filterbank, dB, per-clip standardisation.
    wav: [B, 96000] (or [1, N]); returns ([B, 64, T] standardised spectrogram, filterbank)."""
    spec = m.combined_log_linear_spec
    stft = torch.stft(wav, n_fft=1024, hop_length=320, window=torch.hann_window(1024, device=wav.device), return_complex=True)
    fb = spec._get_current_filter_bank(device=wav.device)
    filtered = torch.matmul(fb, stft.abs())
    db = power_to_db(filtered ** 2)
    mean, std = db.mean(dim=(1, 2), keepdim=True), db.std(dim=(1, 2), keepdim=True) + 1e-5
    return (db - mean) / std, fb


class Core(nn.Module):
    """Everything after the standardised dB spectrogram [B, 64, T]: backbone, GRU, projection, attention, classifier."""

    def __init__(self, m):
        super().__init__()
        self.phi, self.gru, self.projection = m.phi, m.gru, m.projection
        self.attention, self.fc = m.keyword_attention, m.fc

    def forward(self, x):
        x = self.phi(x).permute(0, 2, 1).contiguous()
        x, _ = self.gru(x)
        x = self.projection(x)
        x, _ = self.attention(x)
        return self.fc(x)


def device_forward(m, core, wav):
    """What the device computes, for a batch: per-clip front-end, then the core. Same result batched or one clip at a
    time (unlike m(wav), see power_to_db)."""
    spec, _ = frontend(m, wav)
    return core(spec)
