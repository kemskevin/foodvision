#!/usr/bin/env python
"""Fine-tuning d'un CNN pré-entraîné + export ONNX + quantization INT8
+ génération du jeu de référence pour le drift.

À lancer sur Colab / Kaggle (GPU). Produit un dossier "bundle" :
  model_fp32.onnx, model_int8.onnx, labels.json, reference.csv, metrics.json, samples/

Exemple :
  python train.py --dataset food101 --arch efficientnet_b0 --epochs 3 --out artifacts
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from features import image_features, preprocess_for_model
from quantize import quantize_int8

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_datasets(name: str, root: str):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if name == "food101":
        train = datasets.Food101(root, split="train", transform=train_tf, download=True)
        test = datasets.Food101(root, split="test", transform=eval_tf, download=True)
        raw_test = datasets.Food101(root, split="test", download=False)
    elif name == "pets":
        train = datasets.OxfordIIITPet(root, split="trainval", target_types="category",
                                       transform=train_tf, download=True)
        test = datasets.OxfordIIITPet(root, split="test", target_types="category",
                                      transform=eval_tf, download=True)
        raw_test = datasets.OxfordIIITPet(root, split="test", target_types="category", download=False)
    else:
        raise ValueError(f"Dataset inconnu : {name}")
    return train, test, raw_test, list(train.classes)


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Architecture inconnue : {arch}")
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct1 = correct5 = total = 0
    for x, y in tqdm(loader, desc="evaluation"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
        correct1 += (out.argmax(1) == y).sum().item()
        correct5 += (out.topk(5, 1).indices == y[:, None]).any(1).sum().item()
        total += y.size(0)
    return correct1 / total, correct5 / total


def export_onnx(model, path: Path):
    dummy = torch.randn(1, 3, 224, 224)
    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # anciennes versions de torch sans l'argument dynamo
        torch.onnx.export(model, dummy, str(path), **kwargs)


def softmax(z):
    z = z.astype(np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def benchmark(session, x, n: int = 100) -> float:
    for _ in range(10):
        session.run(None, {"input": x})
    t0 = time.perf_counter()
    for _ in range(n):
        session.run(None, {"input": x})
    return (time.perf_counter() - t0) / n * 1000


def main(args):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    train_ds, test_ds, raw_test, classes = build_datasets(args.dataset, args.data_root)
    if args.max_train_samples:
        idx = random.sample(range(len(train_ds)), min(args.max_train_samples, len(train_ds)))
        train_ds = Subset(train_ds, idx)
    print(f"Train : {len(train_ds)} images | Test : {len(test_ds)} images | {len(classes)} classes")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2,
                             num_workers=args.workers, pin_memory=True)

    model = build_model(args.arch, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    for epoch in range(args.epochs):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        history.append({"epoch": epoch + 1, "train_loss": running / seen,
                        "epoch_time_s": time.time() - t0})
        print(history[-1])

    top1, top5 = evaluate(model, test_loader, device)
    print(f"Test PyTorch : top1={top1:.4f} top5={top5:.4f}")

    # ---------- Export ONNX + quantization ----------
    model.eval().cpu()
    fp32_path, int8_path = out / "model_fp32.onnx", out / "model_int8.onnx"
    export_onnx(model, fp32_path)
    calib_idx = random.sample(range(len(raw_test)), args.calib_samples)
    calib = [preprocess_for_model(raw_test[i][0]) for i in tqdm(calib_idx, desc="calibration")]
    quantize_int8(fp32_path, int8_path, calib, method=args.calib_method)
    (out / "labels.json").write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    # ---------- Jeu de référence pour le drift + comparaison FP32 / INT8 ----------
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
    pd.DataFrame(rows).to_csv(out / "reference.csv", index=False)

    # ---------- Échantillons pour la démo et le test de charge ----------
    for k, i in enumerate(random.sample(range(len(raw_test)), args.n_samples)):
        img, label = raw_test[i]
        img.convert("RGB").save(out / "samples" / f"{k:02d}_{classes[label]}.jpg", quality=90)

    x = calib[0]
    metrics = {
        "test_top1_torch": top1,
        "test_top5_torch": top5,
        "ref_top1_onnx_fp32": ok32 / len(ref_idx),
        "ref_top1_onnx_int8": ok8 / len(ref_idx),
        "latency_ms_fp32": benchmark(sess_fp32, x),
        "latency_ms_int8": benchmark(sess_int8, x),
        "size_mb_fp32": fp32_path.stat().st_size / 1e6,
        "size_mb_int8": int8_path.stat().st_size / 1e6,
    }
    params = {k: str(v) for k, v in vars(args).items()}
    params["num_classes"] = str(len(classes))
    (out / "metrics.json").write_text(
        json.dumps({"params": params, "metrics": metrics, "history": history}, indent=2),
        encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Bundle prêt dans {out.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="food101", choices=["food101", "pets"])
    p.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_large"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--calib-samples", type=int, default=200)
    p.add_argument("--calib-method", default="minmax", choices=["minmax", "percentile", "entropy"])
    p.add_argument("--reference-samples", type=int, default=2000)
    p.add_argument("--n-samples", type=int, default=40)
    main(p.parse_args())
