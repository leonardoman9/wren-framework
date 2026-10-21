"""The device pipeline on the computer (C front-end + int8 TFLite backbone + C tail, the firmware's own C code) on
labelled windows, against the float model (TensorFlow environment).
Usage: python -m wren.tfside.device_eval REF.npz MODEL.tflite LIBFE LIBTAIL K OUT.json
REF.npz: pcm, logits (PyTorch), labels (wren/torchside/windows_ref.py). Reports accuracy and macro-F1 for float and
int8 and the class agreement between them; the package stage gates on the macro-F1 drop."""
import ctypes
import json
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

ref = np.load(sys.argv[1]); model, libfe, libtail, K, out = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
fe_lib, tail_lib = ctypes.CDLL(libfe), ctypes.CDLL(libtail)
fe_lib.BirdFE_SizeOf.restype = ctypes.c_uint32
fe_lib.BirdFE_Init.argtypes = [ctypes.c_void_p]
fe_lib.BirdFE_Push.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
fe_lib.BirdFE_Finish.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int]; fe_lib.BirdFE_Finish.restype = ctypes.c_bool
fe_lib.BirdFE_Reset.argtypes = [ctypes.c_void_p]
fe = ctypes.create_string_buffer(fe_lib.BirdFE_SizeOf())
fe_lib.BirdFE_Init(fe)


class Tail(ctypes.Structure):  # bird_tail.h (hidden size 64, fixed by the architecture)
    _fields_ = [("h", ctypes.c_float * 64), ("ctx", ctypes.c_float * 64), ("mx", ctypes.c_float), ("sm", ctypes.c_float)]


tail_lib.BirdTail_Step.argtypes = [ctypes.POINTER(Tail), ctypes.POINTER(ctypes.c_float)]
tail_lib.BirdTail_Finish.argtypes = [ctypes.POINTER(Tail), ctypes.POINTER(ctypes.c_float)]
tail_lib.BirdTail_Reset.argtypes = [ctypes.POINTER(Tail)]
if tail_lib.BirdTail_Classes() != K: sys.exit(f"the tail library has {tail_lib.BirdTail_Classes()} classes, expected {K}")
it = Interpreter(model_path=model, experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1); it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]
isc, izp = i["quantization"]; osc, ozp = o["quantization"]
got = []
for pcm in ref["pcm"]:
    fe_lib.BirdFE_Reset(fe)
    for off in range(0, len(pcm), 512):
        part = np.ascontiguousarray(pcm[off:off + 512]); fe_lib.BirdFE_Push(fe, part.ctypes.data, len(part))
    q = np.zeros((301, 64), np.int8)
    if not fe_lib.BirdFE_Finish(fe, q.ctypes.data, isc, izp): sys.exit("the C front-end did not finish a window")
    it.set_tensor(i["index"], q.reshape(i["shape"])); it.invoke()
    feats = ((it.get_tensor(o["index"]).reshape(301, 32).astype(np.float32) - ozp) * osc).astype(np.float32)
    t = Tail(); tail_lib.BirdTail_Reset(ctypes.byref(t))
    for row in feats: tail_lib.BirdTail_Step(ctypes.byref(t), row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    lg = (ctypes.c_float * K)(); tail_lib.BirdTail_Finish(ctypes.byref(t), lg); got.append(np.array(lg))
got, lf, y = np.stack(got), ref["logits"], ref["labels"]


def macro(pred):
    f1s = []
    for c in range(K):
        if (y == c).sum() == 0: continue
        tp = ((pred == c) & (y == c)).sum(); p = tp / max(1, (pred == c).sum()); r = tp / max(1, (y == c).sum())
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s))


res = {"windows": int(len(y)), "float_accuracy": float((lf.argmax(1) == y).mean()), "int8_accuracy": float((got.argmax(1) == y).mean()),
       "float_macro_f1": macro(lf.argmax(1)), "int8_macro_f1": macro(got.argmax(1)),
       "class_agreement": float((lf.argmax(1) == got.argmax(1)).mean()), "logits_max_diff": float(np.abs(got - lf).max())}
open(out, "w").write(json.dumps(res, indent=2) + "\n")
print(json.dumps(res))
