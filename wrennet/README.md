# WrenNet

The neural network every wren model uses (constitution III): a learned log-linear spectral front-end, a PhiNet /
MatchboxNet convolutional backbone, a GRU and keyword attention, at 32 kHz on 3 s windows. Only the number of classes,
the learned filterbank parameters and the weights change between projects.

```
models/                 the architecture (Improved_Phi_GRU_ATT and its building blocks)
spectral/               the learned front-end (differentiable_spec.py)
checkpoints/birds.pt    the bird model: 8 species + non-bird, shipped as audiomoth/firmware/bird/model
```

The pipeline imports it as `wrennet` from the repository root (`wren/torchside/model.py`). `checkpoints/birds.pt` is
the starting point of training with `init: wrennet` and the model that `examples/birds/project.yaml` converts.

## Origin

Imported from [sciapponi/WrenNet](https://github.com/sciapponi/WrenNet) at 47c7059, without history. The files in
`models/` and `spectral/` are unchanged from that repository's `birds_distillation_edge/`, so the module names inside
them (`legacy_models`, `legacy_modules`) are WrenNet's. WrenNet's original training, knowledge-distillation and dataset
code is not here: wren trains the model itself (`wren/torchside/train.py`). It is in the WrenNet repository, and in
this repository's history up to commit 7a60637.

## Licence

MIT, as the rest of wren-framework (`LICENSE` at the repository root).
