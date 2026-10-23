# Implementation Plan: Stages re-run when the model code changes

**Branch**: `004-model-code-stage-keys` | **Date**: 2026-10-23 | **Spec**: [spec.md](spec.md)

## Summary

A `model_code()` hash in `wren/runner.py` covers the model's source files only: `wrennet/**/*.py` and wren's loader
`wren/torchside/model.py` (not READMEs, not caches; the checkpoint stays in the keys by content where it is used). It
joins the export and package keys, which lack it, and replaces the train key's hash of the whole `wrennet/` folder, which
also changed on README edits. The package and train keys move into small functions (`package.input_key`,
`data.train_key`) so the stage-key tests can check them, as they already check `export.input_key`.

## Technical Context

**Language/Version**: Python 3.11 (unchanged) · **Dependencies**: none new · **Testing**: `tests/test_stage_keys.py`
(pytest), `tests/test_birds_reproduction.sh` · **Constraints**: no output byte changes (FR-006); keys change once, so
existing runs re-run export, package (and train) once

## Constitution Check

| Principle | Effect | Status |
|-----------|--------|--------|
| IV. Verification | a stale converted model can no longer pass as up to date | Pass (strengthens) |
| V. Reproducibility | keys decide whether a stage runs, not what it produces | Pass |
| Pipeline & Quality Gates ("skipped when its inputs have not changed") | the model code is now an input of every stage that loads the model | Pass (fixes) |

## Project Structure

```text
wren/runner.py            # model_code()
wren/stages/export.py     # input_key: + model_code()
wren/stages/package.py    # input_key(p, export_dir) (moved out of run) + model_code()
wren/stages/data.py       # train_key(...) (moved out of train): model_code() instead of tree_hash(wrennet/)
tests/test_stage_keys.py  # model_code follows .py files, not README; export, package, train keys follow model_code
```

No data model or contracts: internal keys only.
