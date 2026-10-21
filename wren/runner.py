"""Running steps: each in its environment (torch, tf or the host), logged to runs/<name>/logs/, stopping on errors."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from wren.config import ROOT


class StepFailed(SystemExit):
    pass


class Runner:
    def __init__(self, project):
        self.p = project
        self.logs = project.run_dir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def run(self, name, cmd, env=None, cwd=None):
        """Runs cmd, writes its output to logs/<name>.log, returns the output; raises StepFailed on a non-zero exit."""
        e = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                 WRENNET_ROOT=str(ROOT / "third_party" / "WrenNet"), **(env or {}))
        cmd = [str(c) for c in cmd]
        print(f"  - {name}", flush=True)
        r = subprocess.run(cmd, env=e, cwd=cwd, capture_output=True, text=True)
        log = self.logs / f"{name}.log"
        log.write_text("$ " + " ".join(cmd) + "\n" + r.stdout + r.stderr)
        if r.returncode != 0:
            tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
            raise StepFailed(f"step '{name}' failed (exit {r.returncode}); full log: {log}\n{tail}")
        return r.stdout

    def torch(self, name, *args):
        return self.run(name, [self.p.torch_python, *args])

    def tf(self, name, *args):
        return self.run(name, [self.p.tf_python, *args])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_key(*parts):
    """Hash of a stage's inputs: files by content, everything else by value."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, Path) and p.is_file(): h.update(sha256(p).encode())
        else: h.update(json.dumps(p, default=str, sort_keys=True).encode())
    return h.hexdigest()


def listing(*folders):
    """Every file under the folders with its size and modification time: a stage key that notices added, removed or
    changed audio (hashing paths alone would not)."""
    items = []
    for d in folders:
        d = Path(d)
        if d.is_file(): items.append((str(d), d.stat().st_size, int(d.stat().st_mtime)))
        elif d.is_dir():
            items += sorted((str(f), f.stat().st_size, int(f.stat().st_mtime)) for f in d.rglob("*") if f.is_file())
    return items


def code(*names):
    """Hash of wren's own code files, so a changed stage reruns."""
    return [sha256(ROOT / "wren" / n) for n in names]


def submodule_commit(name):
    """Commit of a third_party submodule (the WrenNet architecture, the firmware): part of the keys that depend on it."""
    import subprocess
    return subprocess.run(["git", "-C", str(ROOT / "third_party" / name), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def up_to_date(out_dir, key):
    f = Path(out_dir) / "stage.json"
    return f.exists() and json.loads(f.read_text()).get("key") == key


def mark_done(out_dir, key, **info):
    (Path(out_dir) / "stage.json").write_text(json.dumps({"key": key, **info}, indent=2) + "\n")


def need(path, what):
    if not Path(path).exists(): raise StepFailed(f"{what} not found: {path}")
    return Path(path)


def host_python():
    return sys.executable
