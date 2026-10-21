"""Runs a conversion script with WrenNet's model loader replaced by wren's (any number of classes).
Usage: python -m wren.torchside.run SCRIPT.py [ARGS...]

The scripts import `bird_model_export_check` after putting their own folder (or WRENNET_ROOT/tools) on sys.path;
a module already in sys.modules wins over both, so installing wren's stand-in first is enough.
"""
import runpy
import sys

from wren.torchside import bird_model_export_check

sys.modules["bird_model_export_check"] = bird_model_export_check
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
