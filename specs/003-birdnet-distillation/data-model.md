# Data Model: Learning from BirdNET

## BirdNET installation (`third_party/birdnet/v2.4/`)

| File | Content | Check |
|------|---------|-------|
| `audio-model.tflite` | BirdNET v2.4 GLOBAL 6K audio model, input [1, 144000] float32 at 48 kHz, output [1, 6522] logits | sha256 `55f3e4055b1a13bfa9a2452731d0d34f6a02d6b775a334362665892794165e4c` |
| `labels/en_us.txt` | 6,522 lines `Scientific name_Common name`, in output order | sha256 `b50b77b7c3dfe40cd637e8cccdca0173a0a4ddee8867b830ff3c1a566f477f16` |
| `.wren-birdnet` | the two sha256 above, written last | setup and the project check read it |

## Species match (project check)

| Field | Rule |
|-------|------|
| project sound | every class except the background (`data.negative`) |
| scientific name | `training.birdnet_species[folder]`, else the folder name with `_` → space |
| BirdNET index | the one label line whose scientific name equals it, ignoring case; none → error listing the sound |

## BirdNET grid (`runs/<name>/train/birdnet/grid.json`, `windows.f32`)

- One entry per 3 s window: `[file index, start sample at 32 kHz]` for every training file, starts `0, h, 2h, …` with
  `h = data.windows.hop_seconds × 32000`, up to `length − 96000` (prepare has already looped files shorter than 3 s
  to 3 s, so every file has at least one window).
- `windows.f32`: the windows resampled to 48 kHz, `[n, 144000]` float32 (a work file; may be deleted after teacher.npz).

## Teacher file (`runs/<name>/train/birdnet/teacher.npz`)

| Array | Shape | Meaning |
|-------|-------|---------|
| `grid` | [n, 2] int64 | file index, start sample (as grid.json) |
| `confidence` | [n, S] float32 | sigmoid(logit) of each project species, in class order without the background |
| `top_index` | [n] int32 | BirdNET's most confident label overall (for the report) |
| `top_confidence` | [n] float32 | its confidence |

Plus `stage.json`: `{"key": …}` from prepare's key, the two BirdNET sha256, the species mapping and the grid step.

## Training target (in `train.py`, not stored)

For a drawn window (file fi, start s after jitter): the grid entry of fi nearest to s → c (S values) → p = [c…,
Π(1 − c)] (background last; without background p = c) → floor 1e-4 → normalise → q = p^(1/T), normalised. Classes are
in the order of `classes` (background last, as prepare orders them).

## `train.json` additions

```json
"birdnet": {"model": "BirdNET v2.4 GLOBAL 6K (Zenodo 15050749)", "licence": "CC BY-NC 4.0",
            "weight": 0.5, "temperature": 4.0, "focal_gamma": 2.0, "grid_windows": 2400,
            "agreement": {"<class>": {"windows": 120, "agree": 0.87}},
            "background_with_species": {"windows": 310, "above_0_5": 4}}
```
