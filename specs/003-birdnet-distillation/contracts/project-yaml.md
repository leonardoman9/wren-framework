# Contract: project.yaml options for learning from BirdNET

Training projects only (`data:`); in a convert project (`checkpoint:`) these keys are an error ("only for training").

```yaml
training:
  birdnet: false                 # true: also learn from BirdNET's opinion of each training window
                                 #   (bird species BirdNET knows only; needs python -m wren setup --birdnet)
  birdnet_weight: 0.5            # how much BirdNET counts against your folders, 0-1
  birdnet_temperature: 4         # how much BirdNET's opinion is softened, > 0
  birdnet_species: {}            # optional: folder -> scientific name, when a folder is not named after it
                                 #   e.g. {tawny_owl: Strix aluco}
  focal_gamma: 2.0               # (existing) used by the folder-label part when birdnet is true
```

## Project check (before any long step)

| Condition | Message (plain) |
|-----------|-----------------|
| `birdnet: true`, BirdNET not installed and no stored BirdNET results for this project (`runs/<name>/train/birdnet/mapping.json`) | "Learning from BirdNET needs BirdNET on this computer: run python -m wren setup --birdnet" |
| BirdNET not installed, stored results exist but do not cover a current sound folder | "BirdNET's stored results do not cover 'x': run python -m wren setup --birdnet" |
| a sound is not a BirdNET species | "Learning from BirdNET works only for bird species BirdNET knows. Not found: 'upsweep', 'trill'. Name each folder after its scientific name, or give it under training: birdnet_species (e.g. upsweep: Turdus merula)." |
| `birdnet: true` and `loss: ce` | "Learning from BirdNET uses the focal loss; remove 'loss: ce' (or set birdnet: false)." |
| `birdnet_weight` outside 0–1, `birdnet_temperature` ≤ 0 | "training.birdnet_weight: a number from 0 to 1" / "training.birdnet_temperature: a number above 0" |
| `birdnet_species` names a folder that is not a sound, or the background | "training.birdnet_species: 'x' is not one of your sound folders" |

With `birdnet: false` (default) none of these run and BirdNET is never mentioned (FR-008).

Without BirdNET installed, species matching uses the stored `mapping.json` instead of the labels file (spec SC-006).
The train stage then reuses the stored results if their key matches the current training files; otherwise it stops
before training: "BirdNET is needed for the new training audio: run python -m wren setup --birdnet".
