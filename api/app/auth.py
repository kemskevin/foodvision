import hashlib
import secrets

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from .config import settings
from .db import SessionLocal, User


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def generate_token() -> str:
    return "sk_" + secrets.token_urlsafe(32)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_user(x_api_key: str | None = Header(default=None),
                 db: Session = Depends(get_db)) -> User:
    """Vérifie le header X-API-Key. Les tokens sont stockés hachés (SHA-256)."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Header X-API-Key manquant")
    user = db.query(User).filter(User.token_hash == hash_token(x_api_key), User.active.is_(True)).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Clé API invalide ou révoquée")
    return user


def require_admin(x_admin_token: str | None = Header(default=None)):
    if not x_admin_token or not secrets.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(status_code=403, detail="Token admin invalide")
