"""Chargement du modèle ONNX depuis le Model Registry MLflow (alias) ou un dossier local,
avec un thread qui recharge automatiquement le modèle quand l'alias change."""
import json
import logging
import tempfile
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import settings
from .features import preprocess_for_model

log = logging.getLogger("model")


class ModelNotReady(RuntimeError):
    pass


class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._session = None
        self._input_name = "input"
        self._labels: list[str] = []
        self.version = "none"
        self.source = "none"
        self._stop = threading.Event()
        self._thread = None

    @property
    def ready(self) -> bool:
        return self._session is not None

    # ---------- MLflow ----------
    def _client(self):
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_uri)
        return mlflow.MlflowClient()

    def _lookup_alias(self):
        """Retourne (alias, version) du premier alias existant, sinon None."""
        if not settings.mlflow_uri:
            return None
        client = self._client()
        for alias in dict.fromkeys([settings.model_alias, settings.fallback_alias]):
            if not alias:
                continue
            try:
                mv = client.get_model_version_by_alias(settings.model_name, alias)
                return alias, mv.version
            except Exception:
                continue
        return None

    def _resolve_bundle(self):
        found = None
        try:
            found = self._lookup_alias()
        except Exception as exc:
            log.warning("MLflow injoignable : %s", exc)
        if found:
            import mlflow

            alias, version = found
            dst = Path(tempfile.gettempdir()) / "model_cache" / f"{settings.model_name}_v{version}"
            if not any(dst.rglob("labels.json")):
                dst.mkdir(parents=True, exist_ok=True)
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{settings.model_name}/{version}", dst_path=str(dst))
            return dst, f"v{version}", f"mlflow:{alias}"
        local = Path(settings.local_model_dir)
        if local.exists() and any(local.rglob("labels.json")):
            return local, "local", "local"
        raise ModelNotReady("Aucun modèle trouvé (ni dans MLflow, ni dans le dossier local)")

    @staticmethod
    def _find(bundle: Path, name: str) -> Path:
        matches = sorted(bundle.rglob(name))
        if not matches:
            raise ModelNotReady(f"{name} introuvable dans {bundle}")
        return matches[0]

    # ---------- Chargement / prédiction ----------
    def load(self) -> str:
        bundle, version, source = self._resolve_bundle()
        model_path = self._find(bundle, f"model_{settings.model_variant}.onnx")
        labels = json.loads(self._find(bundle, "labels.json").read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = settings.ort_threads
        session = ort.InferenceSession(str(model_path), sess_options=opts,
                                       providers=["CPUExecutionProvider"])
        with self._lock:
            self._session = session
            self._input_name = session.get_inputs()[0].name
            self._labels = labels
            self.version = version
            self.source = source
        log.info("Modèle chargé : %s (%s, %s, %d classes)", version, source, model_path.name, len(labels))
        return version

    def predict(self, img, top_k: int = 5):
        with self._lock:
            session, name, labels, version = self._session, self._input_name, self._labels, self.version
        if session is None:
            raise ModelNotReady("Modèle non chargé")
        logits = session.run(None, {name: preprocess_for_model(img)})[0][0].astype(np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx = np.argsort(probs)[::-1][:top_k]
        return [{"label": labels[i], "score": round(float(probs[i]), 4)} for i in idx], version

    # ---------- Rechargement automatique ----------
    def _watch(self):
        while not self._stop.wait(settings.refresh_seconds):
            try:
                found = self._lookup_alias()
                if found and f"v{found[1]}" != self.version:
                    log.info("Nouvelle version détectée pour l'alias %s : v%s", *found)
                    self.load()
            except Exception as exc:
                log.warning("Vérification MLflow échouée : %s", exc)

    def start_watcher(self):
        if settings.mlflow_uri and settings.refresh_seconds > 0 and self._thread is None:
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()

    def stop_watcher(self):
        self._stop.set()


model_manager = ModelManager()
