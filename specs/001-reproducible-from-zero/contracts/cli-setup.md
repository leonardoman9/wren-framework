# Contract: `python -m wren setup`

## Synopsis

```text
python -m wren setup [--tflm] [--check [--verbose]] [--python PATH]
```

| Flag | Meaning |
|---|---|
| (none) | Checks, then creates or updates `.venv-torch` and `.venv-tf` from the platform's lock files |
| `--tflm` | Also fetches TFLM at the pinned commit, applies the patch, and builds it (this downloads GCC 14.3.1) |
| `--check` | Changes nothing; reports the state of every item in plain words ("Analysis tools: ready", "Firmware build tools: missing — run python -m wren setup --tflm") and exits non-zero if anything is missing or wrong |
| `--verbose` | With `--check`: also prints component names and versions (Python packages, TFLM commit, GCC version), for developers |
| `--python PATH` | Python 3.11 interpreter used to create the environments (default: the current one if it is 3.11, else `python3.11` on PATH) |

## Preconditions

- It runs on any Python 3.x with the standard library only: it imports nothing outside the stdlib, and
  `wren/__main__.py → cli.py` must not import `yaml`, `numpy` or the stages before dispatching to `setup`.
- The current directory does not matter. Paths resolve from the repository root.

## Checks (in order; the first failure stops the run)

| # | Check | Failure message (user-facing, no jargon) |
|---|---|---|
| 1 | Platform is Linux x86_64, or macOS arm64 at version 14 or later (the pinned PyTorch has wheels for macOS 14+ only) | macOS x86_64: "This Mac has an Intel processor. The analysis libraries no longer support it directly: please use Docker (see README, 'Using Docker')." Other: the same, naming the platform. |
| 2 | Python 3.11 available | "Python 3.11 is needed (found 3.X). Install it from python.org or with your package manager, then run this again." |
| 3 | Free disk ≥ 6 GB in the repo's filesystem (measured need ~4 GB + 30 %; `WREN_MIN_DISK_GB` may lower it, CI only) | "About 6 GB of free disk space is needed; X GB are free." |
| 4 | Tools: git, make, cc, clang (firmware simulator with sanitizers), wget, mtools, qemu-system-arm; on macOS also `gmake` (GNU make 4: macOS's make 3.81 cannot build TFLM) | One line per missing tool, with the `apt install …` / `brew install …` line for the current OS |
| 5 | Lock files exist for this platform and match their `.in` | Developer-facing: "envs/lock/… is out of date with envs/requirements-….in: regenerate the locks (see envs/README)" |

## Actions

1. For each environment: if it is a symlink, remove the link (not its target). If it is missing, or its
   `.wren-lock` marker differs from the lock hash, create the venv and run
   `pip install --require-hashes --no-deps -r envs/lock/<env>-<platform>.txt`, then write the marker.
   Otherwise skip it.
2. With `--tflm`: run `audiomoth/scripts/setup_tflm.sh`, then `HOST=1 scripts/build_tflm.sh`,
   unless both libraries exist and TFLM HEAD is e0021ca. Verify that GCC reports 14.3.1.
3. Print one plain line ("Ready. Next: python -m wren run your-project.yaml"). Component names and versions (TFLM, GCC, package versions) are written only to `runs/_setup/logs/setup.log` and shown with `--check --verbose` (Principle I: no technical jargon in user-facing output).

## Output and exit codes

| Exit | Meaning |
|---|---|
| 0 | Ready (or `--check`: everything OK) |
| 1 | An install step failed (the pip or build log is kept in `runs/_setup/logs/`, and the message points to it) |
| 2 | A precondition check failed (see table) |

Running it twice in a row is idempotent: the second run performs no downloads or installs and exits 0.

## `python -m wren run` and the environments

`run` needs PyYAML and the pipeline's packages, which live in `.venv-torch`. If the typed Python lacks them, `run`
re-executes itself as `.venv-torch/bin/python -m wren run …` with the same arguments. If `.venv-torch` does not exist
yet, it stops with "Not installed yet: run `python -m wren setup --tflm` first." Users therefore type the same
`python -m wren …` for both commands (constitution II).
