# Python environments

The pipeline runs in two environments, because PyTorch and TensorFlow need different `numpy` versions:

| Environment | Intent (top-level pins) | Lock files (every package, with hashes) |
|---|---|---|
| `.venv-torch` | `requirements-torch.in` | `lock/torch-linux-x86_64.txt`, `lock/torch-macos-arm64.txt` |
| `.venv-tf` | `requirements-tf.in` | `lock/tf-linux-x86_64.txt`, `lock/tf-macos-arm64.txt` |

`python -m wren setup` installs from the lock of the current platform with
`pip install --require-hashes --no-deps`, and so does the Dockerfile (Linux locks). Users never edit these files.

## Changing a dependency (developers)

1. Edit `requirements-<env>.in`. Keep exact pins (`==`), because the int8 conversion is version-sensitive.
2. Run `envs/lock.sh` (needs [uv](https://docs.astral.sh/uv/)). It rewrites all four locks, starting from the current
   ones, so the versions of packages you did not touch stay as they are (to move them on purpose, delete the lock
   first). Each lock starts with `# input-sha256:` of its `.in` file, and setup refuses a lock that no longer matches
   its `.in`.
3. Run `tests/test_birds_reproduction.sh` in the Docker image. The bird package must stay byte-identical, or
   the change must come with a new reference.

Notes:
- Linux uses the CPU-only PyTorch wheels (`download.pytorch.org/whl/cpu`), so no CUDA packages are installed.
- The macOS locks target macOS 14 or later, the oldest version with wheels for the pinned PyTorch.

## CI-only setting

`WREN_MIN_DISK_GB` lowers the free-disk threshold that `python -m wren setup` requires. It is for CI runners
with a smaller disk only, not for users.
