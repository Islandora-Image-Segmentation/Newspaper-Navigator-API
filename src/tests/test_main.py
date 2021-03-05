import sys
import os

from fastapi.testclient import TestClient
from PIL import Image

from . import TEST_ASSETS_DIR
from main import app
from utils import image_to_base64


client = TestClient(app)


def test_base64_gibberish():
    base64 = "this is not valid base64"
    response = client.post("/api/segment_base64", json={"image_base64": base64})
    assert response.json()["status_code"] == -1
    assert response.json()["segment_count"] is None


def test_base64_shapes():
    image = Image.open(os.path.join(TEST_ASSETS_DIR, "shapes.png"))
    base64 = image_to_base64(image)
    response = client.post("/api/segment_base64", json={"image_base64": base64})
    assert response.json()["status_code"] == 0
    assert response.json()["segment_count"] == 0


def test_base64_newspaper():
    image = Image.open(os.path.join(TEST_ASSETS_DIR, "newspaper_issue.png"))
    base64 = image_to_base64(image)
    response = client.post("/api/segment_base64", json={"image_base64": base64})
    assert response.json()["status_code"] == 0
    assert response.json()["segment_count"] > 0
