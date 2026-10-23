"""Runs a conversion script with WrenNet's model loader replaced by wren's (any number of classes).
Usage: python -m wren.torchside.run SCRIPT.py [ARGS...]

The scripts (export_backbone.py, make_specs.py, copied from WrenNet's tools/) import `bird_model_export_check`;
a module already in sys.modules wins over any file of that name, so installing wren's stand-in first is enough.
"""
import runpy
import sys

from wren.torchside import bird_model_export_check

sys.modules["bird_model_export_check"] = bird_model_export_check
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
