# Data Model: Reproducible from Zero

The feature has no database. Its "entities" are files with fixed formats, plus the states the setup moves
through.

## Calibration bundle

`examples/birds/calibration/`: 3 s clips from freely licensed xeno-canto recordings (the model's 8 species first,
then other European species from a fixed list), used to quantise, evaluate and test the bird model.

| Path | Format | Rule |
|---|---|---|
| `audio/*.flac` | FLAC, PCM_16, mono, 32 kHz, 3.0 s each | At least `calibration.count + calibration.eval` clips (400 by default); only xeno-canto recordings licensed CC0, CC-BY or CC-BY-SA; no BioDCASE |
| `SHA256SUMS` | `sha256sum` format | One line per clip, checked by the reproduction test |
| `ATTRIBUTION.csv` | CSV | One row per clip: `clip,xc_id,species,recordist,licence,licence_url,source_url,start_s`; every clip in `audio/` has exactly one row |
| `LICENSE.md` | text | Each clip keeps its original CC licence (see `ATTRIBUTION.csv`); the clips are modified (cut, resampled, re-encoded) |
| `README.md` | text | Source (xeno-canto API v3), selection rule, how the clips were cut |

**Relationships**:
- `examples/birds/project.yaml` points to it with `calibration.audio: [calibration/audio]` and
  `test.audio: [calibration/audio]`.
- The export stage takes the first `count + eval` clips after the seed-0 shuffle. The package stage takes
  `test.windows` clips after the seed-7 shuffle.

**Validation**:
- Existing: `wren.config` fails if a listed path does not exist.
- New: a unit test fails if any YAML under `examples/` or `tests/` has an absolute path, or mentions BioDCASE.
- New: a unit test fails if `ATTRIBUTION.csv` and `audio/` do not match one to one, or if any `licence` is not
  CC0, CC-BY or CC-BY-SA.

## Lock file

`envs/lock/<env>-<platform>.txt`, with `<env>` ∈ {torch, tf} and `<platform>` ∈ {linux-x86_64, macos-arm64}.

| Field | Rule |
|---|---|
| header comment | The `uv pip compile` command and the `.in` file hash it was generated from |
| one line per package | `name==version` plus `--hash=sha256:…` (all transitive packages) |

**Validation**: setup refuses a lock whose header hash does not match the current `.in` file. The message
asks for the locks to be regenerated, and it is shown to developers only, because users never edit `.in`
files.

## Environment

`.venv-torch/`, `.venv-tf/`: real directories, never symlinks.

| Attribute | Rule |
|---|---|
| interpreter | Python 3.11.x |
| `.wren-lock` marker | sha256 of the lock file it was installed from; a mismatch means it gets reinstalled |

## Toolchain and TFLM

| Item | Location | Identity check |
|---|---|---|
| TFLM source | `audiomoth/third_party/tflite-micro` (git-ignored) | `git rev-parse HEAD` = e0021ca434cd…, patch applied |
| ARM GCC | `…/tflite-micro/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/` | `arm-none-eabi-gcc --version` contains `14.3.1` |
| TFLM libraries | `…/tflite-micro/gen/cortex_m_generic_cortex-m4+fp_{release,release_with_logs}_cmsis_nn_gcc/lib/libtensorflow-microlite.a` | file exists |

## Reproduction reference

`tests/expected/birds_firmware.sha256`, in `sha256sum` format:

```text
<sha256>  birds.bin
<sha256>  birds_diag.bin
```

It is recorded from the Docker reference path. The package reference is
`audiomoth/firmware/bird/model/`, regenerated from the calibration bundle (FR-015).

## Run provenance (report addition)

`runs/<name>/report/` gains a `provenance` block:

| Field | Value |
|---|---|
| `path` | `docker` or `native` |
| `image` | Docker image ID (from `/proc/self/cgroup` or the `WREN_IMAGE_ID` build arg), `null` when native |
| `platform` | e.g. `linux-x86_64`, `macos-arm64` |
| `locks` | sha256 of the lock files used |
| `sources` | git commit of wren-framework (+ `dirty` flag); content hashes of `wrennet/` and `audiomoth/` |
| `byte_reproducible` | `true` only for `docker` |

## Setup states

```text
start
  └─ checks (python, platform, disk, tools) ── fail → stop with plain message (exit 2)
       └─ environments (per env: missing | symlink | stale lock → (re)create; ok → skip)
            └─ [--tflm] TFLM (missing → fetch+patch; not built → build; gcc ≠ 14.3.1 → error)
                 └─ ready (exit 0)
```

`--check` runs every state without changing anything and reports one line per item.
