#!/usr/bin/env python
"""Déplace un alias MLflow vers une version donnée (ex : promotion du canary en production).
Les API détectent le changement automatiquement (MODEL_REFRESH_SECONDS).

  python training/promote.py --tracking-uri http://localhost:5000 --version 2 --alias production
"""
import argparse

import mlflow
from mlflow import MlflowClient

p = argparse.ArgumentParser()
p.add_argument("--tracking-uri", default="http://localhost:5000")
p.add_argument("--name", default="food-classifier")
p.add_argument("--version", required=True)
p.add_argument("--alias", default="production")
args = p.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)
MlflowClient().set_registered_model_alias(args.name, args.alias, args.version)
print(f"Alias '{args.alias}' -> {args.name} v{args.version}")
