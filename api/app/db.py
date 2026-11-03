import logging
import time
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import settings

log = logging.getLogger("db")

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


def utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UsageLog(Base):
    """Une ligne par appel : sert au suivi d'usage par utilisateur ET au drift."""
    __tablename__ = "usage_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    endpoint: Mapped[str] = mapped_column(String(50))
    status_code: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    deployment: Mapped[str] = mapped_column(String(20))
    model_version: Mapped[str] = mapped_column(String(50))
    predicted_class: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    brightness: Mapped[float | None] = mapped_column(Float, nullable=True)
    contrast: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpness: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


def init_db(retries: int = 15):
    """Crée les tables (avec retry : Postgres peut démarrer après l'API)."""
    for attempt in range(1, retries + 1):
        try:
            Base.metadata.create_all(engine)
            return
        except (OperationalError, ProgrammingError) as exc:  # DB pas prête ou course entre réplicas
            log.warning("DB pas prête (%s/%s) : %s", attempt, retries, exc.__class__.__name__)
            time.sleep(2)
    raise RuntimeError("Impossible d'initialiser la base de données")
