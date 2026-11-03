"""Test de charge de l'API.

  API_TOKEN=sk_... locust -f loadtest/locustfile.py --host http://IP:8080 \
      --headless -u 50 -r 5 -t 3m --html loadtest/report.html

DRIFT_MODE=1 envoie des images assombries et floutées pour déclencher la détection de drift.
"""
import glob
import io
import os
import random

from locust import HttpUser, between, events, task
from PIL import Image, ImageEnhance, ImageFilter

TOKEN = os.getenv("API_TOKEN", "")
SAMPLES_DIR = os.getenv("SAMPLES_DIR", "models/samples")
DRIFT_MODE = os.getenv("DRIFT_MODE", "0") == "1"


def load_images() -> list[bytes]:
    paths = [p for ext in ("jpg", "jpeg", "png") for p in glob.glob(f"{SAMPLES_DIR}/*.{ext}")]
    images = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        if DRIFT_MODE:
            img = ImageEnhance.Brightness(img).enhance(0.35).filter(ImageFilter.GaussianBlur(3))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        images.append(buf.getvalue())
    return images


IMAGES = load_images()


@events.test_start.add_listener
def check(environment, **kwargs):
    if not IMAGES:
        raise RuntimeError(f"Aucune image trouvée dans {SAMPLES_DIR}")
    if not TOKEN:
        print("ATTENTION : API_TOKEN vide, toutes les requêtes renverront 401")
    print(f"{len(IMAGES)} images chargées (drift_mode={DRIFT_MODE})")


class ApiUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(10)
    def predict(self):
        self.client.post(
            "/predict",
            files={"file": ("img.jpg", random.choice(IMAGES), "image/jpeg")},
            headers={"X-API-Key": TOKEN},
            name="/predict",
        )

    @task(1)
    def health(self):
        self.client.get("/health")
