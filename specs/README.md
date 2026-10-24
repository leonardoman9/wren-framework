# Feature specifications

Each folder is one feature, made with Spec Kit (spec → plan → tasks → implementation → pull request) under the
project constitution (`.specify/memory/constitution.md`).

| Feature | State | Pull request |
|---------|-------|--------------|
| 001 Reproducible from zero | merged | #1 |
| 002 Clean up `wrennet/` and `audiomoth/` | merged | #4 |
| 003 Learning from BirdNET for bird projects | merged | #5 |
| 004 Stages re-run when the model code changes | merged | #6 |
| 006 Re-measure the bird model's accuracy | merged | #7 |
| 005 Drop unused packages from the torch environment | planned (spec, plan, tasks) | — |
| 008 Fix the BirdNET target and measure whether it helps | specified | — |
| 009 Training experiments, then deploy the best | specified | — |
| 007 The wren-framework manual | specified; last, it describes the result of all the others | — |

Order of the queue: 005 → 008 → 009 → 007. Each feature's own branch carries its implementation; the folders here
are the agreed plan.
