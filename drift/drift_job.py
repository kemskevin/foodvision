#!/usr/bin/env python
"""Détection de data drift avec Evidently.

Compare les features des images reçues en production (table usage_logs)
au jeu de référence (reference.csv, généré à l'entraînement et versionné avec le modèle dans MLflow).
Produit des rapports HTML + JSON et expose des métriques Prometheus.

  python drift_job.py          # boucle infinie (service)
  python drift_job.py --once   # une seule analyse
"""
import argparse
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from prometheus_client import Gauge, start_http_server
from sqlalchemy import create_engine, text

try:
    from evidently import ColumnMapping
except ImportError:  # selon la version d'evidently 0.4.x
    from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s drift %(message)s")
log = logging.getLogger("drift")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local.db")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MODEL_NAME = os.getenv("MODEL_NAME", "food-classifier")
REF_ALIAS = os.getenv("REFERENCE_ALIAS", "production")
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "/models")
REPORT_DIR = Path(os.getenv("REPORT_DIR", "/reports"))
INTERVAL = int(os.getenv("DRIFT_INTERVAL_SECONDS", "300"))
WINDOW_HOURS = int(os.getenv("DRIFT_WINDOW_HOURS", "24"))
MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "50"))
DEPLOYMENT = os.getenv("DRIFT_DEPLOYMENT", "stable")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))
KEEP_REPORTS = int(os.getenv("KEEP_REPORTS", "50"))

NUM = ["brightness", "contrast", "sharpness", "confidence"]
CAT = ["predicted_class"]

G_DRIFT = Gauge("drift_dataset_drift", "1 si drift détecté sur le dataset, 0 sinon")
G_SHARE = Gauge("drift_share_of_drifted_columns", "Part des features en drift")
G_SAMPLES = Gauge("drift_current_samples", "Nombre de prédictions analysées")
G_COLUMN = Gauge("drift_column_score", "Score de drift par feature (p-value ou distance)", ["column"])
G_LAST = Gauge("drift_last_run_timestamp", "Timestamp de la dernière analyse")


def load_reference():
    if MLFLOW_URI:
        try:
            import mlflow

            mlflow.set_tracking_uri(MLFLOW_URI)
            mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, REF_ALIAS)
            dst = Path(tempfile.gettempdir()) / f"reference_v{mv.version}"
            if not any(dst.rglob("reference.csv")):
                dst.mkdir(parents=True, exist_ok=True)
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{MODEL_NAME}/{mv.version}", dst_path=str(dst))
            return pd.read_csv(next(dst.rglob("reference.csv"))), f"v{mv.version}"
        except Exception as exc:
            log.warning("Référence MLflow indisponible (%s), essai du dossier local", exc)
    local = Path(LOCAL_MODEL_DIR)
    if local.exists():
        found = sorted(local.rglob("reference.csv"))
        if found:
            return pd.read_csv(found[0]), "local"
    return None, None


def load_current(engine) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)
    query = text(
        "SELECT predicted_class, confidence, brightness, contrast, sharpness "
        "FROM usage_logs WHERE status_code = 200 AND deployment = :dep AND created_at >= :since"
    )
    return pd.read_sql(query, engine, params={"dep": DEPLOYMENT, "since": since})


def cleanup_reports():
    reports = sorted(REPORT_DIR.glob("drift_*.html"))
    for old in reports[:-KEEP_REPORTS]:
        old.unlink(missing_ok=True)


def run_once(engine) -> dict | None:
    ref, ref_version = load_reference()
    if ref is None:
        log.warning("Aucun jeu de référence trouvé")
        return None
    cur = load_current(engine)
    G_SAMPLES.set(len(cur))
    if len(cur) < MIN_SAMPLES:
        log.info("Seulement %d prédictions (< %d) : analyse reportée", len(cur), MIN_SAMPLES)
        return None

    cols = NUM + CAT
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref[cols], current_data=cur[cols],
               column_mapping=ColumnMapping(numerical_features=NUM, categorical_features=CAT))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_DIR / f"drift_{ts}.html"))
    report.save_html(str(REPORT_DIR / "latest.html"))

    result = report.as_dict()
    dataset = result["metrics"][0]["result"]
    by_col = result["metrics"][1]["result"].get("drift_by_columns", {})
    summary = {
        "timestamp": ts,
        "reference_version": ref_version,
        "current_samples": len(cur),
        "dataset_drift": bool(dataset.get("dataset_drift")),
        "share_of_drifted_columns": float(dataset.get("share_of_drifted_columns", 0.0)),
        "columns": {c: {"drift_detected": bool(v.get("drift_detected")),
                        "score": float(v.get("drift_score", 0.0)),
                        "test": v.get("stattest_name")} for c, v in by_col.items()},
    }
    (REPORT_DIR / "latest.json").write_text(json.dumps(summary, indent=2))
    with open(REPORT_DIR / "history.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    cleanup_reports()

    G_DRIFT.set(1 if summary["dataset_drift"] else 0)
    G_SHARE.set(summary["share_of_drifted_columns"])
    for c, v in summary["columns"].items():
        G_COLUMN.labels(c).set(v["score"])
    G_LAST.set(time.time())

    level = logging.WARNING if summary["dataset_drift"] else logging.INFO
    log.log(level, "Drift=%s | part de features en drift=%.0f%% | n=%d",
            summary["dataset_drift"], summary["share_of_drifted_columns"] * 100, len(cur))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    if args.once:
        summary = run_once(engine)
        print(json.dumps(summary, indent=2) if summary else "Pas d'analyse (données insuffisantes)")
        return
    start_http_server(METRICS_PORT)
    while True:
        try:
            run_once(engine)
        except Exception:
            log.exception("Échec de l'analyse de drift")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
