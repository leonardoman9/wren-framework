"""Stand-in for WrenNet's tools/bird_model_export_check.py, whose load_model() is fixed at 9 classes.

The conversion scripts wren/torchside/export_backbone.py and make_specs.py, copied from WrenNet, do
`from bird_model_export_check import SR, frontend, load_model`;
wren/torchside/run.py installs this module under that name first, so they get the general loader unchanged.
"""
from wren.torchside.model import CLIP_S, SR, Core, frontend, load_model  # noqa: F401
