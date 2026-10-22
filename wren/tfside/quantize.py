"""Int8 TFLite backbone from the onnx2tf SavedModel (runs in the TensorFlow environment).
Usage: python -m wren.tfside.quantize SAVED_MODEL_DIR FLOAT32.tflite SPECS.npz N_CALIB OUT.tflite REPORT.json

The recipe of the shipped bird model (WrenNet's tools/bird_conversion/quant_variants.py, int8 mode): full-integer
quantization with int8 input and output, calibrated on the first N_CALIB spectrograms of SPECS.npz 'calib'; then the
tensor names are stripped (smaller flash). The report compares the int8 backbone with the float one on 'eval'.
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.tools import flatbuffer_utils

saved, f32_path, specs_path, n_cal, out, report = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
specs = np.load(specs_path)
calib, ev = specs["calib"][:n_cal], specs["eval"]
MIN_CALIB = 20  # below this the int8 ranges are guesswork; the int8-vs-float gate still guards everything above it
if len(calib) < MIN_CALIB:
    sys.exit(f"only {len(calib)} windows of audio to tune the compressed model, at least {MIN_CALIB} are needed: add recordings")
if len(calib) < n_cal:  # a small dataset: use all it has (calibration.count is an upper bound)
    print(f"note: {len(calib)} calibration spectrograms available, {n_cal} requested; using all {len(calib)}")
IN_SHAPE = tuple(tf.saved_model.load(saved).signatures["serving_default"].structured_input_signature[1]["spec"].shape)
conv = tf.lite.TFLiteConverter.from_saved_model(saved)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = conv.inference_output_type = tf.int8


def rep():
    for c in calib:
        yield [c.reshape(IN_SHAPE).astype(np.float32)]


conv.representative_dataset = rep
raw = out + ".unstripped"
open(raw, "wb").write(conv.convert())


def run(path, x, quant):
    it = Interpreter(model_path=path); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    outs = []
    for s in x:
        v = s.reshape(IN_SHAPE)
        if quant:
            sc, zp = i["quantization"]; v = np.clip(np.rint(v / sc) + zp, -128, 127).astype(np.int8)
        it.set_tensor(i["index"], v); it.invoke()
        y = it.get_tensor(o["index"]).reshape(301, 32)
        if quant:
            sc, zp = o["quantization"]; y = (y.astype(np.float32) - zp) * sc
        outs.append(y)
    return np.stack(outs)


f32, q = run(f32_path, ev, False), run(raw, ev, True)
rel = float(np.sqrt(((q - f32) ** 2).mean() / (f32 ** 2).mean()))
m = flatbuffer_utils.read_model(raw)
names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
ops = {}
for op in m.subgraphs[0].operators:
    n = names[m.operatorCodes[op.opcodeIndex].builtinCode]
    ops[n] = ops.get(n, 0) + 1
flatbuffer_utils.strip_strings(m)
flatbuffer_utils.write_model(m, out)
os.remove(raw)
i = Interpreter(model_path=out).get_input_details()[0]
info = {"calibration": len(calib), "calibration_requested": n_cal, "eval": len(ev), "int8_vs_float_rel_rms": rel, "operators": ops,
        "tensors": len(m.subgraphs[0].tensors), "bytes": os.path.getsize(out),
        "input_scale": float(i["quantization"][0]), "input_zero_point": int(i["quantization"][1])}
open(report, "w").write(json.dumps(info, indent=2) + "\n")
print(json.dumps(info))
unsupported = set(ops) - {"ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "HARD_SWISH", "LOGISTIC", "MEAN", "MUL", "PAD"}
if unsupported: sys.exit(f"operators the firmware's resolver does not register: {sorted(unsupported)}")
