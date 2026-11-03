import io
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, File, HTTPException, Query, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field
from sqlalchemy import case, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .auth import generate_token, get_db, hash_token, require_admin, require_user
from .config import settings
from .db import UsageLog, User, init_db
from .features import image_features
from .model import ModelNotReady, model_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

PRED_TOTAL = Counter("api_predictions_total", "Nombre de prédictions",
                     ["deployment", "model_version", "status"])
PRED_LATENCY = Histogram("api_prediction_latency_seconds", "Latence de prédiction",
                         ["deployment", "model_version"],
                         buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
PRED_CONF = Histogram("api_prediction_confidence", "Confiance top-1",
                      ["deployment", "model_version"],
                      buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        model_manager.load()
    except Exception as exc:
        log.warning("Modèle non chargé au démarrage : %s", exc)
    model_manager.start_watcher()
    yield
    model_manager.stop_watcher()


app = FastAPI(title="FoodVision API", version="1.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class UserCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")


def _log_usage(db: Session, user: User, endpoint: str, status: int, latency_ms: float,
               version: str, pred: dict | None = None, feats: dict | None = None):
    try:
        db.add(UsageLog(
            user_id=user.id, endpoint=endpoint, status_code=status, latency_ms=latency_ms,
            deployment=settings.deployment, model_version=version,
            predicted_class=pred["label"] if pred else None,
            confidence=pred["score"] if pred else None,
            **(feats or {}),
        ))
        db.commit()
    except Exception:
        db.rollback()
        log.exception("Échec de l'enregistrement de l'usage")


# ---------------------------------------------------------------- public
@app.get("/health")
def health():
    return {"status": "ok", "deployment": settings.deployment, "model_loaded": model_manager.ready,
            "model_version": model_manager.version, "model_source": model_manager.source}


@app.post("/predict")
def predict(response: Response,
            file: UploadFile = File(...),
            top_k: int = Query(5, ge=1, le=10),
            user: User = Depends(require_user),
            db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    version = model_manager.version

    def fail(status: int, detail: str):
        _log_usage(db, user, "/predict", status, (time.perf_counter() - t0) * 1000, version)
        PRED_TOTAL.labels(settings.deployment, version, str(status)).inc()
        raise HTTPException(status_code=status, detail=detail)

    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        fail(413, f"Image trop volumineuse (max {settings.max_upload_mb} Mo)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        fail(400, "Fichier image invalide")
    try:
        preds, version = model_manager.predict(img, top_k)
    except ModelNotReady:
        fail(503, "Modèle non chargé")

    feats = image_features(img)
    latency_ms = (time.perf_counter() - t0) * 1000
    _log_usage(db, user, "/predict", 200, latency_ms, version, preds[0], feats)
    PRED_TOTAL.labels(settings.deployment, version, "200").inc()
    PRED_LATENCY.labels(settings.deployment, version).observe(latency_ms / 1000)
    PRED_CONF.labels(settings.deployment, version).observe(preds[0]["score"])

    response.headers["X-Model-Version"] = version
    response.headers["X-Deployment"] = settings.deployment
    return {"predictions": preds, "model_version": version,
            "deployment": settings.deployment, "latency_ms": round(latency_ms, 1)}


@app.get("/me/usage")
def my_usage(user: User = Depends(require_user), db: Session = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    total = db.query(func.count(UsageLog.id)).filter(UsageLog.user_id == user.id).scalar()
    last_24h = db.query(func.count(UsageLog.id)).filter(
        UsageLog.user_id == user.id, UsageLog.created_at >= since).scalar()
    return {"user": user.name, "total_requests": total, "requests_last_24h": last_24h}


# ---------------------------------------------------------------- admin
@app.post("/admin/users", status_code=201, dependencies=[Depends(require_admin)])
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    token = generate_token()
    user = User(name=body.name, token_hash=hash_token(token))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Utilisateur déjà existant")
    return {"name": user.name, "api_key": token,
            "warning": "Conservez cette clé : elle ne sera plus jamais affichée."}


@app.get("/admin/users", dependencies=[Depends(require_admin)])
def list_users(db: Session = Depends(get_db)):
    return [{"name": u.name, "active": u.active, "created_at": u.created_at}
            for u in db.query(User).order_by(User.id)]


@app.delete("/admin/users/{name}", dependencies=[Depends(require_admin)])
def revoke_user(name: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.name == name).first()
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur inconnu")
    user.active = False
    db.commit()
    return {"name": name, "active": False}


@app.get("/admin/usage", dependencies=[Depends(require_admin)])
def usage(hours: int = Query(24, ge=1, le=24 * 90), db: Session = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = (
        db.query(
            User.name.label("name"),
            func.count(UsageLog.id).label("requests"),
            func.sum(case((UsageLog.status_code >= 400, 1), else_=0)).label("errors"),
            func.avg(UsageLog.latency_ms).label("avg_latency_ms"),
            func.max(UsageLog.created_at).label("last_call"),
        )
        .join(UsageLog, UsageLog.user_id == User.id)
        .filter(UsageLog.created_at >= since)
        .group_by(User.name)
        .order_by(func.count(UsageLog.id).desc())
        .all()
    )
    return {"window_hours": hours, "users": [
        {"user": r.name, "requests": r.requests, "errors": int(r.errors or 0),
         "avg_latency_ms": round(float(r.avg_latency_ms or 0), 1), "last_call": r.last_call}
        for r in rows]}


@app.post("/admin/reload", dependencies=[Depends(require_admin)])
def reload_model():
    try:
        return {"model_version": model_manager.load(), "deployment": settings.deployment}
    except ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc))
