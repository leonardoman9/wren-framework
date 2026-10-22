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
        "name": "keys", "checkpoint": str(ROOT / "wrennet" / "best_distillation_model.pt"),
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
