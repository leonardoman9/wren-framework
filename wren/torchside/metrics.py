"""Classification metrics (no sklearn needed)."""
import numpy as np


def confusion(y_true, y_pred, k):
    m = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred): m[int(t), int(p)] += 1
    return m


def scores(y_true, y_pred, k, names=None):
    """Accuracy, macro-F1 over classes present in y_true, per-class precision/recall/F1, confusion matrix."""
    m = confusion(y_true, y_pred, k)
    tp = np.diag(m).astype(float)
    prec = np.divide(tp, m.sum(0), out=np.zeros(k), where=m.sum(0) > 0)
    rec = np.divide(tp, m.sum(1), out=np.zeros(k), where=m.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    present = m.sum(1) > 0
    names = names or [str(i) for i in range(k)]
    return {
        "accuracy": float(tp.sum() / max(1, m.sum())),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {names[i]: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                                 "f1": round(float(f1[i]), 4), "support": int(m.sum(1)[i])} for i in range(k)},
        "confusion": m.tolist(),
    }
