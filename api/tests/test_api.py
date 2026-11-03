"""Tests de l'API avec un mini-modèle ONNX factice (pas besoin du vrai modèle)."""
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image

TMP = Path(tempfile.mkdtemp())
N_CLASSES = 5


def _build_dummy_model(path: Path):
    """input (N,3,224,224) -> GlobalAveragePool -> Flatten -> Gemm -> logits (N,5)."""
    w = np.random.RandomState(0).randn(N_CLASSES, 3).astype(np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["input"], ["gap"]),
         helper.make_node("Flatten", ["gap"], ["flat"]),
         helper.make_node("Gemm", ["flat", "W", "B"], ["logits"], transB=1)],
        "dummy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 224, 224])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", N_CLASSES])],
        [helper.make_tensor("W", TensorProto.FLOAT, w.shape, w.flatten()),
         helper.make_tensor("B", TensorProto.FLOAT, b.shape, b)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


_build_dummy_model(TMP / "model_int8.onnx")
(TMP / "labels.json").write_text(json.dumps([f"class_{i}" for i in range(N_CLASSES)]))

os.environ.update({
    "DATABASE_URL": f"sqlite:///{TMP / 'test.db'}",
    "ADMIN_TOKEN": "test-admin",
    "MLFLOW_TRACKING_URI": "",
    "LOCAL_MODEL_DIR": str(TMP),
})

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ADMIN = {"X-Admin-Token": "test-admin"}


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 260), (200, 120, 40)).save(buf, "JPEG")
    return buf.getvalue()


def test_full_flow():
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is True

        # sans token -> 401
        r = c.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 401
        # mauvais token -> 401
        r = c.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")},
                   headers={"X-API-Key": "sk_faux"})
        assert r.status_code == 401
        # admin sans token admin -> 403
        assert c.post("/admin/users", json={"name": "alice"}).status_code == 403

        r = c.post("/admin/users", json={"name": "alice"}, headers=ADMIN)
        assert r.status_code == 201
        key = {"X-API-Key": r.json()["api_key"]}
        assert c.post("/admin/users", json={"name": "alice"}, headers=ADMIN).status_code == 409

        r = c.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}, headers=key)
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["predictions"]) == 5
        assert r.headers["X-Model-Version"] == "local"

        r = c.post("/predict", files={"file": ("a.txt", b"pas une image", "text/plain")}, headers=key)
        assert r.status_code == 400

        usage = c.get("/admin/usage", headers=ADMIN).json()["users"]
        assert usage[0]["user"] == "alice"
        assert usage[0]["requests"] == 2
        assert usage[0]["errors"] == 1
        assert c.get("/me/usage", headers=key).json()["total_requests"] == 2

        # révocation
        assert c.delete("/admin/users/alice", headers=ADMIN).status_code == 200
        r = c.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}, headers=key)
        assert r.status_code == 401

        assert "api_predictions_total" in c.get("/metrics/").text
