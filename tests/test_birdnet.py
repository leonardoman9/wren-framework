"""Learning from BirdNET (specs/003-birdnet-distillation): the bird-species check of the project (contracts/project-yaml.md)
and the teacher target and loss (contracts/teacher-target.md). Nothing is downloaded: BirdNET is replaced by a small
species list in a temporary folder.
Usage: .venv-torch/bin/python -m pytest tests/test_birdnet.py"""
import json

import pytest
import yaml

from wren import birdnet, config

LABELS = ["Bubo bubo_Eurasian Eagle-Owl", "Strix aluco_Tawny Owl", "Periparus ater_Coal Tit", "Apus apus_Common Swift",
          "Dog_Dog", "Engine_Engine", "Poecile montanus_Willow Tit", "Emberiza cia_Rock Bunting",
          "Certhia familiaris_Eurasian Treecreeper", "Certhia brachydactyla_Short-toed Treecreeper",
          "Lophophanes cristatus_Crested Tit", "Turdus merula_Eurasian Blackbird"]


@pytest.fixture
def fake_birdnet(tmp_path, monkeypatch):
    d = tmp_path / "birdnet"
    (d / "labels").mkdir(parents=True)
    (d / "labels" / "en_us.txt").write_text("\n".join(LABELS))
    (d / "audio-model.tflite").write_bytes(b"not a model")
    (d / birdnet.MARKER).write_text(json.dumps(birdnet.FILES))
    monkeypatch.setattr(birdnet, "DIR", d)
    return d


def project(tmp_path, folders, training, negative="background"):
    data = tmp_path / "data"
    for f in folders: (data / f).mkdir(parents=True, exist_ok=True)
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "name": "birdnet-check", "data": {"root": "data", "negative": negative}, "training": training,
        "runs": str(tmp_path / "runs")}))
    return tmp_path / "project.yaml"


def refusal(path):
    with pytest.raises(SystemExit) as e:
        config.load(path)
    return str(e.value)


# ---------------------------------------------------------------- species matching

def test_match_by_scientific_name(fake_birdnet):
    names = birdnet.labels()
    matched, unmatched = birdnet.match(["Bubo_bubo", "periparus ater", "tawny_owl", "upsweep", "background"], "background",
                                       {"tawny_owl": "Strix aluco"}, names)
    assert matched == {"Bubo_bubo": {"name": "Bubo bubo", "index": 0}, "periparus ater": {"name": "Periparus ater", "index": 2},
                       "tawny_owl": {"name": "Strix aluco", "index": 1}}
    assert unmatched == ["upsweep"]  # the background is never matched


def test_bird_project_accepted(tmp_path, fake_birdnet):
    p = config.load(project(tmp_path, ["Bubo_bubo", "Apus_apus", "background"], {"birdnet": True}))
    assert p.training["birdnet"] and p.training["loss"] == "focal"
    assert p.training["birdnet_weight"] == 0.5 and p.training["birdnet_temperature"] == 4.0
    assert p.training["birdnet_species"] == {"Apus_apus": {"name": "Apus apus", "index": 3},
                                             "Bubo_bubo": {"name": "Bubo bubo", "index": 0}}


def test_sounds_that_are_not_birds_refused(tmp_path, fake_birdnet):
    msg = refusal(project(tmp_path, ["upsweep", "downsweep", "trill", "background"], {"birdnet": True}))
    assert "only for bird species BirdNET knows" in msg
    for f in ("upsweep", "downsweep", "trill"): assert f"'{f}'" in msg
    assert "birdnet_species" in msg and "background" not in msg.split("Not found:")[1].split(".")[0]


def test_birdnet_species_names_a_folder(tmp_path, fake_birdnet):
    p = config.load(project(tmp_path, ["tawny_owl", "Bubo_bubo", "background"],
                            {"birdnet": True, "birdnet_species": {"tawny_owl": "Strix aluco"}}))
    assert p.training["birdnet_species"]["tawny_owl"] == {"name": "Strix aluco", "index": 1}
    msg = refusal(project(tmp_path, ["Bubo_bubo", "background"], {"birdnet": True, "birdnet_species": {"owl": "Strix aluco"}}))
    assert "'owl' is not one of your sound folders" in msg


@pytest.mark.parametrize("training, expected", [
    ({"birdnet": True, "loss": "ce"}, "uses the focal loss"),
    ({"birdnet": True, "birdnet_weight": 1.5}, "birdnet_weight: a number from 0 to 1"),
    ({"birdnet": True, "birdnet_temperature": 0}, "birdnet_temperature: a number above 0"),
])
def test_option_values(tmp_path, fake_birdnet, training, expected):
    assert expected in refusal(project(tmp_path, ["Bubo_bubo", "background"], training))


def test_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(birdnet, "DIR", tmp_path / "nothing")
    assert "run python -m wren setup --birdnet" in refusal(project(tmp_path, ["Bubo_bubo", "background"], {"birdnet": True}))


def test_option_off_needs_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(birdnet, "DIR", tmp_path / "nothing")
    p = config.load(project(tmp_path, ["upsweep", "background"], {"epochs": 3}))
    assert not any(k.startswith("birdnet") for k in p.training)  # training as before this feature (SC-005)
    assert "BirdNET" not in capsys.readouterr().out


def test_convert_project_refuses_the_option(tmp_path):
    (tmp_path / "m.pt").write_bytes(b"")
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "checkpoint": "m.pt", "classes": ["a", "b"], "calibration": {"audio": ["m.pt"]}, "training": {"birdnet": True}}))
    assert "only for training projects" in refusal(tmp_path / "project.yaml")


def stored(tmp_path, species):
    d = tmp_path / "runs" / "birdnet-check" / "train" / "birdnet"
    d.mkdir(parents=True)
    (d / "mapping.json").write_text(json.dumps({"birdnet_sha256": birdnet.FILES, "species": species}))


def test_stored_results_replace_birdnet(tmp_path, monkeypatch):
    """Another computer, without BirdNET, repeats a training from a copied run folder (spec SC-006)."""
    monkeypatch.setattr(birdnet, "DIR", tmp_path / "nothing")
    stored(tmp_path, {"Bubo_bubo": {"name": "Bubo bubo", "index": 0}, "Apus_apus": {"name": "Apus apus", "index": 3}})
    p = config.load(project(tmp_path, ["Bubo_bubo", "Apus_apus", "background"], {"birdnet": True}))
    assert p.training["birdnet_species"]["Apus_apus"]["index"] == 3


def test_stored_results_must_cover_every_sound(tmp_path, monkeypatch):
    monkeypatch.setattr(birdnet, "DIR", tmp_path / "nothing")
    stored(tmp_path, {"Bubo_bubo": {"name": "Bubo bubo", "index": 0}})
    msg = refusal(project(tmp_path, ["Bubo_bubo", "Apus_apus", "background"], {"birdnet": True}))
    assert "stored results do not cover 'Apus_apus'" in msg
