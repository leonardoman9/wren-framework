# Quickstart: validating "Reproducible from Zero"

Each scenario maps to a spec story. Commands run from the repository root of a fresh
`git clone` (no submodules).

## Prerequisites

- **Docker path**: Docker only.
- **Native path**: Python 3.11, git, make, a C compiler, wget, mtools and qemu-system-arm, plus 15 GB free.
  See the [setup contract](contracts/cli-setup.md) for the per-OS install lines.

## 1. Clean machine, native (US1, SC-001, SC-005)

```sh
python3.11 -m wren setup --tflm
python3.11 -m wren setup --check        # every line OK, exit 0
python3.11 -m wren run examples/birds/project.yaml
```

**Expected**:
- `runs/birds/firmware/birds.bin` exists.
- Every parity check passes, and the report names the run "native, verified by parity checks".
- No step outside these commands is needed.
- Running `setup` a second time downloads nothing.

## 2. Clean machine, Docker (US1, US3, SC-004)

```sh
docker build --platform linux/amd64 -f docker/Dockerfile -t wren .
docker run --rm --platform linux/amd64 -v "$PWD/runs:/wren/runs" --entrypoint tests/test_birds_reproduction.sh wren
```

**Expected**: `identical` for all 7 package files, the firmware sha256 values match
`tests/expected/birds_firmware.sha256`, and the run ends with `PASS`. Repeating this on a second host (macOS
and Linux) gives the same result.

## 3. Bring-your-own data (US4)

```sh
python tests/make_synthetic_dataset.py runs/synth_data           # stands in for a user's class folders
python -m wren run tests/synthetic/project.yaml  # near-minimal YAML (data, negative, a few training overrides)
```

**Expected**: the train stage passes the evaluate→export gate, the firmware is built, and the report is
written. Every unspecified parameter takes its documented default.

## 4. CI (US2, SC-002, SC-003)

Open a pull request. **Expected**: the `docker`, `native-linux` and `native-macos` jobs are green, each in
under 20 minutes with warm caches (see the [CI contract](contracts/ci-workflow.md)).

## 5. Failure paths (spec edge cases)

| Do | Expected |
|---|---|
| Run `python3.12 -m wren setup` with no 3.11 installed | Exit 2: "Python 3.11 is needed (found 3.12)…" |
| Replace `.venv-tf` with a symlink, then `python -m wren setup` | The link is removed, a real venv is created, and the link target is untouched |
| Run setup on an Intel Mac | Exit 2 with the Docker pointer, before any download |
| Delete or alter one clip in `examples/birds/calibration/audio/`, then run `tests/test_birds_reproduction.sh` | It fails at the `SHA256SUMS` check, naming the clip, before any stage runs |
| Add an absolute path or a BioDCASE path to any YAML under `examples/` or `tests/` | `tests/test_examples.py` fails |
