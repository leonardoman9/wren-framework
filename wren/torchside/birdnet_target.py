"""The training target from BirdNET's opinion and the focal distillation loss (specs/003-birdnet-distillation,
contracts/teacher-target.md). Torch only; used by wren/torchside/train.py.

The loss is the original WrenNet project's FocalDistillationLoss, (1 - alpha) * focal + alpha * T^2 * KL, with its
teacher target corrected: BirdNET's confidences are independent values in [0, 1], not a distribution, so they become
one (the project's species plus a background share for "none of them") before the temperature softens them."""
import torch
import torch.nn.functional as F

FLOOR = 1e-9     # no class gets probability 0 (log 0 in the KL term); about the smallest confidence BirdNET gives
SILENT = 0.1     # a species window where BirdNET gives every project species less than this: its part is skipped


def focal(logits, y, gamma):
    """Mean focal loss; gamma 0 is plain cross-entropy (the training's folder-label loss)."""
    ce = F.cross_entropy(logits, y, reduction="none")
    return (ce if gamma == 0 else (1 - torch.exp(-ce)) ** gamma * ce).mean()


def target(confidence, negative, classes, temperature):
    """confidence [B, S]: BirdNET's sigmoid confidence for each project species, in class order without the background.
    Returns q [B, classes]: p = species confidences plus the background share prod(1 - c) at the background's index
    (none when negative < 0), floored and normalised, then softened: q = p^(1/T), normalised."""
    c = confidence.clamp(0.0, 1.0)
    if negative >= 0:
        none = torch.prod(1.0 - c, dim=1, keepdim=True)
        p = torch.cat([c[:, :negative], none, c[:, negative:]], dim=1)
    else:
        p = c
    assert p.shape[1] == classes, (p.shape, classes)
    p = p.clamp_min(FLOOR)
    p = p / p.sum(1, keepdim=True)
    q = p ** (1.0 / temperature)
    return q / q.sum(1, keepdim=True)


def mask(confidence, y, negative):
    """1 where BirdNET's part counts; 0 on a species window where BirdNET hears none of the project's species (a faint or
    distant call keeps its folder label instead of being pulled towards background)."""
    silent = confidence.max(1).values < SILENT
    return (~(silent & (y != negative))).float()


def focal_distillation(logits, y, q, m, alpha, temperature, gamma):
    """(1 - alpha) * focal(folder labels) + alpha * T^2 * KL(q || softmax(logits / T)), the KL averaged over the windows
    with m = 1."""
    hard = focal(logits, y, gamma)
    log_s = F.log_softmax(logits / temperature, dim=1)
    kl = (q * (torch.log(q) - log_s)).sum(1)
    soft = (kl * m).sum() / m.sum().clamp_min(1.0)
    return (1.0 - alpha) * hard + alpha * temperature ** 2 * soft
