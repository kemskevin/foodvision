#!/usr/bin/env python
"""Refait la quantization INT8 SANS ré-entraîner, à partir de model_fp32.onnx.

Teste plusieurs stratégies, garde celle qui conserve le mieux la précision,
régénère reference.csv (drift) et met à jour metrics.json.

  python requantize.py --bundle artifacts --dataset food101
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from torchvision import datasets
from tqdm import tqdm

from features import image_features, preprocess_for_model
from quantize import quantize_dynamic_int8, quantize_int8

CONV_OPS = ["Conv", "MatMul", "Gemm"]
CONFIGS = {
    "static_minmax_all": dict(kind="static", method="minmax", op_types=None),
    "static_minmax_conv": dict(kind="static", method="minmax", op_types=CONV_OPS),
    "static_percentile_conv": dict(kind="static", method="percentile", op_types=CONV_OPS),
    "dynamic": dict(kind="dynamic"),
}


def load_raw_test(name: str, root: str):
    if name == "food101":
        return datasets.Food101(root, split="test", download=True)
    return datasets.OxfordIIITPet(root, split="test", target_types="category", download=True)


def make_session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def softmax(z):
    z = z.astype(np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def accuracy(session, samples):
    ok = 0
    for x, label in samples:
        ok += int(session.run(None, {"input": x})[0][0].argmax() == label)
    return ok / len(samples)


def benchmark(session, x, n: int = 100) -> float:
    for _ in range(10):
        session.run(None, {"input": x})
    t0 = time.perf_counter()
    for _ in range(n):
        session.run(None, {"input": x})
    return (time.perf_counter() - t0) / n * 1000


def main(args):
    bundle = Path(args.bundle)
    fp32_path = bundle / "model_fp32.onnx"
    classes = json.loads((bundle / "labels.json").read_text(encoding="utf-8"))
    raw_test = load_raw_test(args.dataset, args.data_root)

    random.seed(123)
    idx = random.sample(range(len(raw_test)), args.calib_samples + args.eval_samples)
    calib = [preprocess_for_model(raw_test[i][0]) for i in tqdm(idx[:args.calib_samples], desc="calibration")]
    evalset = [(preprocess_for_model(raw_test[i][0]), raw_test[i][1])
               for i in tqdm(idx[args.calib_samples:], desc="jeu d'évaluation")]

    sess_fp32 = make_session(fp32_path)
    acc_fp32 = accuracy(sess_fp32, evalset)
    lat_fp32 = benchmark(sess_fp32, calib[0])
    print(f"\nFP32 : précision={acc_fp32:.3f}  latence={lat_fp32:.1f} ms")

    results = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        path = bundle / f"candidate_{name}.onnx"
        try:
            if cfg["kind"] == "static":
                quantize_int8(fp32_path, path, calib, method=cfg["method"], op_types=cfg["op_types"])
            else:
                quantize_dynamic_int8(fp32_path, path)
            sess = make_session(path)
            results[name] = {
                "accuracy": accuracy(sess, evalset),
                "latency_ms": benchmark(sess, calib[0]),
                "size_mb": path.stat().st_size / 1e6,
            }
            print(f"{name:26s} précision={results[name]['accuracy']:.3f}  "
                  f"latence={results[name]['latency_ms']:.1f} ms  taille={results[name]['size_mb']:.1f} Mo")
        except Exception as exc:
            print(f"{name:26s} ÉCHEC : {exc}")

    if not results:
        raise SystemExit("Aucune stratégie n'a fonctionné")

    best = max(results, key=lambda k: (round(results[k]["accuracy"], 2), -results[k]["latency_ms"]))
    print(f"\n==> Meilleure stratégie : {best}")
    shutil.copy(bundle / f"candidate_{best}.onnx", bundle / "model_int8.onnx")
    for p in bundle.glob("candidate_*.onnx"):
        p.unlink()

    # ---------- Jeu de référence pour le drift (avec le nouveau modèle INT8) ----------
    sess_int8 = make_session(bundle / "model_int8.onnx")
    ref_idx = random.sample(range(len(raw_test)), min(args.reference_samples, len(raw_test)))
    rows, ok32, ok8 = [], 0, 0
    for i in tqdm(ref_idx, desc="reference"):
        img, label = raw_test[i]
        x = preprocess_for_model(img)
        p32 = softmax(sess_fp32.run(None, {"input": x})[0][0])
        p8 = softmax(sess_int8.run(None, {"input": x})[0][0])
        ok32 += int(p32.argmax() == label)
        ok8 += int(p8.argmax() == label)
        rows.append({**image_features(img),
                     "predicted_class": classes[int(p8.argmax())],
                     "confidence": float(p8.max()),
                     "true_label": classes[label]})
    pd.DataFrame(rows).to_csv(bundle / "reference.csv", index=False)

    # ---------- Mise à jour de metrics.json ----------
    metrics_path = bundle / "metrics.json"
    info = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    info.setdefault("params", {})["quantization"] = best
    info.setdefault("metrics", {}).update({
        "ref_top1_onnx_fp32": ok32 / len(ref_idx),
        "ref_top1_onnx_int8": ok8 / len(ref_idx),
        "latency_ms_fp32": lat_fp32,
        "latency_ms_int8": results[best]["latency_ms"],
        "size_mb_fp32": fp32_path.stat().st_size / 1e6,
        "size_mb_int8": (bundle / "model_int8.onnx").stat().st_size / 1e6,
    })
    info["quantization_comparison"] = {"fp32": {"accuracy": acc_fp32, "latency_ms": lat_fp32}, **results}
    metrics_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info["metrics"], indent=2))
    print("\nTableau comparatif (pour la slide 8) :")
    print(json.dumps(info["quantization_comparison"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="artifacts")
    p.add_argument("--dataset", default="food101", choices=["food101", "pets"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--calib-samples", type=int, default=200)
    p.add_argument("--eval-samples", type=int, default=500)
    p.add_argument("--reference-samples", type=int, default=2000)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    main(p.parse_args())
