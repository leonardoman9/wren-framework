"""A stage is skipped only when its inputs did not change (constitution, Pipeline & Quality Gates): the export key
must follow the calibration audio by content, not by path, and the conversion code.
Usage: .venv-torch/bin/python -m pytest tests/test_stage_keys.py"""
from pathlib import Path

import yaml

from wren import config, runner
from wren.stages import export

ROOT = Path(__file__).resolve().parents[1]


def project(tmp_path):
    audio = tmp_path / "calibration"
    audio.mkdir()
    (audio / "a.wav").write_bytes(b"one clip")
    (audio / "b.wav").write_bytes(b"another clip")
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "name": "keys", "checkpoint": str(ROOT / "wrennet" / "checkpoints" / "birds.pt"),
        "classes": [f"class {i}" for i in range(9)], "negative": "last",
        "calibration": {"audio": [str(audio)], "count": 1, "eval": 1}, "runs": str(tmp_path / "runs")}))
    return config.load(tmp_path / "project.yaml"), audio


def test_export_key_follows_audio_content(tmp_path):
    p, audio = project(tmp_path)
    before = export.input_key(p)
    assert export.input_key(p) == before  # same inputs, same key
    (audio / "a.wav").write_bytes(b"one clip, re-recorded")  # same path, new content
    assert export.input_key(p) != before


def test_export_key_follows_conversion_code(tmp_path, monkeypatch):
    p, _ = project(tmp_path)
    before = export.input_key(p)
    real = runner.sha256
    monkeypatch.setattr(export, "code", lambda *names: [real(ROOT / "wren" / n) + ("x" if n == "tfside/quantize.py" else "")
                                                         for n in names])
    assert export.input_key(p) != before


# The model code (specs/004-model-code-stage-keys): every stage that loads the model re-runs when it changes.

def test_model_code_follows_source_not_docs(tmp_path):
    for rel in ("wrennet/models/core.py", "wrennet/README.md", "wren/torchside/model.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((ROOT / rel).read_bytes())
    before = runner.model_code(tmp_path)
    (tmp_path / "wrennet/README.md").write_text("edited")
    (tmp_path / "wrennet/__pycache__").mkdir()
    (tmp_path / "wrennet/__pycache__/core.cpython-311.pyc").write_bytes(b"cache")
    assert runner.model_code(tmp_path) == before  # documentation and caches do not count (FR-003)
    (tmp_path / "wrennet/models/core.py").write_text((ROOT / "wrennet/models/core.py").read_text() + "\n# edited\n")
    assert runner.model_code(tmp_path) != before
    (tmp_path / "wrennet/models/core.py").write_bytes((ROOT / "wrennet/models/core.py").read_bytes())
    (tmp_path / "wren/torchside/model.py").write_text("# another loader\n")
    assert runner.model_code(tmp_path) != before


def test_export_package_train_keys_follow_model_code(tmp_path, monkeypatch):
    from wren.stages import data, package
    p, _ = project(tmp_path)
    keys = lambda: (export.input_key(p), package.input_key(p, tmp_path / "export"), data.train_key("prep", {"epochs": 1}))
    before = keys()
    for module in (export, package, data):
        monkeypatch.setattr(module, "model_code", lambda: "another model")
    after = keys()
    assert all(a != b for a, b in zip(before, after))  # FR-001, FR-002, the train key's model part (FR-005)


def test_prepare_key_ignores_model_code():
    import inspect
    from wren.stages import data
    assert "model_code" not in inspect.getsource(data.prepare)  # prepare does not load the model (FR-004)
