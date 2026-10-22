"""Example and test projects stay portable and clean (FR-013, FR-014): no absolute paths, no BioDCASE, and the bird
calibration bundle carries one attribution row per clip, under a free licence only.
Usage: .venv-torch/bin/python -m pytest tests/test_examples.py"""
import csv
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "examples" / "birds" / "calibration"
FREE = ("publicdomain/zero", "licenses/by/", "licenses/by-sa/")
YAMLS = sorted(p for d in ("examples", "tests") for p in (ROOT / d).rglob("*.yaml"))


def strings(node):
    if isinstance(node, dict): return [s for v in node.values() for s in strings(v)]
    if isinstance(node, list): return [s for v in node for s in strings(v)]
    return [node] if isinstance(node, str) else []


@pytest.mark.parametrize("path", YAMLS, ids=lambda p: str(p.relative_to(ROOT)))
def test_yaml_portable(path):
    text = path.read_text()
    assert "biodcase" not in text.lower(), "BioDCASE must not appear anywhere in the project"
    absolute = [s for s in strings(yaml.safe_load(text)) if s.startswith("/") or s.startswith("~") or re.match(r"[A-Za-z]:\\", s)]
    assert not absolute, f"absolute paths: {absolute}"


@pytest.mark.skipif(not BUNDLE.exists(), reason="calibration bundle not built yet")
def test_bundle_attribution():
    rows = list(csv.DictReader((BUNDLE / "ATTRIBUTION.csv").open()))
    clips = {p.name for p in (BUNDLE / "audio").glob("*.flac")}
    listed = [r["clip"] for r in rows]
    assert len(listed) == len(set(listed)), "a clip is listed twice"
    assert set(listed) == clips, f"unlisted: {sorted(clips - set(listed))[:5]}, missing: {sorted(set(listed) - clips)[:5]}"
    bad = [(r["clip"], r["licence_url"]) for r in rows
           if not any(f in r["licence_url"] for f in FREE) or "-nc" in r["licence_url"] or "-nd" in r["licence_url"]]
    assert not bad, f"not CC0 / CC-BY / CC-BY-SA: {bad[:5]}"
