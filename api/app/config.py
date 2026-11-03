import os


class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./local.db")
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    model_name: str = os.getenv("MODEL_NAME", "food-classifier")
    model_alias: str = os.getenv("MODEL_ALIAS", "production")
    fallback_alias: str = os.getenv("MODEL_FALLBACK_ALIAS", "production")
    model_variant: str = os.getenv("MODEL_VARIANT", "int8")  # int8 | fp32
    local_model_dir: str = os.getenv("LOCAL_MODEL_DIR", "/models")
    refresh_seconds: int = int(os.getenv("MODEL_REFRESH_SECONDS", "60"))
    ort_threads: int = int(os.getenv("ORT_THREADS", "0"))
    admin_token: str = os.getenv("ADMIN_TOKEN", "change-me")
    deployment: str = os.getenv("DEPLOYMENT", "stable")  # stable | canary
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "5"))


settings = Settings()
