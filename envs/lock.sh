#!/bin/sh
# Regenerates the lock files in envs/lock/ from envs/requirements-*.in (developers only; needs uv).
# Each lock starts with the sha256 of its .in file, which `python -m wren setup` checks.
set -e
cd "$(dirname "$0")"
for env in torch tf; do
  for plat in linux-x86_64:x86_64-unknown-linux-gnu macos-arm64:aarch64-apple-darwin; do
    name=${plat%%:*}; triple=${plat##*:}
    extra=""
    # Linux: CPU-only torch wheels (no CUDA packages), like the device pipeline itself
    if [ "$env" = torch ] && [ "$name" = linux-x86_64 ]; then
      extra="--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match"
    fi
    # macOS: the pinned torch has wheels for macOS 14+ only
    [ "$name" = macos-arm64 ] && export MACOSX_DEPLOYMENT_TARGET=14.0 || unset MACOSX_DEPLOYMENT_TARGET
    out=lock/$env-$name.txt
    # start from the current lock: uv keeps the versions it finds in the output file, so a change to the .in moves
    # only what it must (a fresh compile would move every unpinned dependency to its newest version)
    [ -f "$out" ] && cp "$out" "$out.tmp"
    # shellcheck disable=SC2086
    uv pip compile requirements-$env.in --python-version 3.11 --python-platform "$triple" --generate-hashes \
      --no-header --quiet --emit-index-url $extra -o "$out.tmp"
    { echo "# input-sha256: $(shasum -a 256 requirements-$env.in | cut -d' ' -f1)"
      echo "# uv pip compile requirements-$env.in --python-version 3.11 --python-platform $triple --generate-hashes $extra"
      cat "$out.tmp"; } > "$out"
    rm "$out.tmp"
    echo "$out: $(grep -c '==' "$out") packages"
  done
done
