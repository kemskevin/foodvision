"""Prétraitement et features d'image partagés entre l'entraînement et l'API.

IMPORTANT : ce fichier est dupliqué dans api/app/features.py.
Les deux copies doivent rester identiques, sinon le drift sera faussé.
"""
import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def resize_center_crop(img: Image.Image, resize: int = 256, crop: int = 224) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = resize / min(w, h)
    nw, nh = max(crop, round(w * scale)), max(crop, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - crop) // 2, (nh - crop) // 2
    return img.crop((left, top, left + crop, top + crop))


def preprocess_for_model(img: Image.Image) -> np.ndarray:
    """Image PIL -> tenseur (1, 3, 224, 224) float32 normalisé ImageNet."""
    x = np.asarray(resize_center_crop(img), dtype=np.float32) / 255.0
    x = (x.transpose(2, 0, 1) - MEAN) / STD
    return x[None].astype(np.float32)


def image_features(img: Image.Image) -> dict:
    """Statistiques simples utilisées pour la détection de data drift."""
    g = np.asarray(resize_center_crop(img).convert("L"), dtype=np.float32) / 255.0
    laplacian = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return {
        "brightness": float(g.mean()),
        "contrast": float(g.std()),
        "sharpness": float(laplacian.var()),
    }
