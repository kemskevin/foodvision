#!/usr/bin/env python
"""Enregistre un bundle (sortie de train.py) dans le Model Registry MLflow
et lui attribue un alias (production ou canary).

  python training/register.py --bundle models --tracking-uri http://localhost:5000 --alias production
"""
import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient


class OnnxImageClassifier(mlflow.pyfunc.PythonModel):
    """Wrapper pyfunc : permet aussi de charger le modèle avec mlflow.pyfunc.load_model."""

    def load_context(self, context):
        import onnxruntime as ort

        self.session = ort.InferenceSession(context.artifacts["model_int8"],
                                            providers=["CPUExecutionProvider"])

    def predict(self, context, model_input, params=None):
        return self.session.run(None, {"input": model_input})[0]


def main(args):
    bundle = Path(args.bundle)
    required = ["model_fp32.onnx", "model_int8.onnx", "labels.json", "reference.csv"]
    missing = [f for f in required if not (bundle / f).exists()]
    if missing:
        raise SystemExit(f"Fichiers manquants dans {bundle} : {missing}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    info = {}
    if (bundle / "metrics.json").exists():
        info = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))

    with mlflow.start_run(run_name=args.run_name) as run:
        if info.get("params"):
            mlflow.log_params(info["params"])
        if info.get("metrics"):
            mlflow.log_metrics(info["metrics"])
        for h in info.get("history", []):
            mlflow.log_metric("train_loss", h["train_loss"], step=h["epoch"])
        artifacts = {
            "model_fp32": str(bundle / "model_fp32.onnx"),
            "model_int8": str(bundle / "model_int8.onnx"),
            "labels": str(bundle / "labels.json"),
            "reference": str(bundle / "reference.csv"),
        }
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=OnnxImageClassifier(),
            artifacts=artifacts,
            registered_model_name=args.name,
            pip_requirements=["onnxruntime", "numpy"],
        )
        run_id = run.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{args.name}'")
    version = max((v for v in versions if v.run_id == run_id), key=lambda v: int(v.version)).version
    client.set_registered_model_alias(args.name, args.alias, version)
    print(f"Modèle {args.name} version {version} enregistré avec l'alias '{args.alias}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="models")
    p.add_argument("--tracking-uri", default="http://localhost:5000")
    p.add_argument("--experiment", default="food-classifier")
    p.add_argument("--name", default="food-classifier")
    p.add_argument("--alias", default="production")
    p.add_argument("--run-name", default="register-bundle")
    main(p.parse_args())
