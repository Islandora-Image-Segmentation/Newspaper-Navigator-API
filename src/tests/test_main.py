import os
import sys
from PIL import Image
from fastapi.testclient import TestClient
from main import app
from utils import image_to_base64

from . import TEST_ASSETS_DIR, TESTS_DIR


client = TestClient(app) #Creates a fake API app for testing.


def test_base64_gibberish():
    """ End-to-end test for the base64 endpoint. Attempts to give non-valid base64. """
    base64 = "this is not valid base64"
    response = client.post("/api/segment_base64", json={"image_base64": base64})
    assert response.json()["status_code"] != 0
    assert response.json()["segment_count"] is None


def test_base64_shapes():
    """ End-to-end test for the base64 endpoint. Attempts to give image without segments. """
    image = Image.open(os.path.join(TEST_ASSETS_DIR, "shapes.png"))
    base64 = image_to_base64(image)
    response = client.post("/api/segment_base64", json={"image_base64": base64})
    assert response.json()["status_code"] == 0
    assert response.json()["segment_count"] == 0


def test_base64_newspaper():
    """ End-to-end test for the base64 endpoint. Attempts to give a valid newspaper. """
    image = Image.open(os.path.join(TEST_ASSETS_DIR, "newspaper_issue.png"))
    base64 = image_to_base64(image)
    response = client.post("/api/segment_base64", json={"image_base64": base64})
    assert response.json()["status_code"] == 0
    assert response.json()["segment_count"] > 0


def test_url_invalid():
    """ End-to-end test for the URL endpoint. Attempts to give an invalid URL. """
    url = "this is not a valid url"
    response = client.post("/api/segment_url", json={"image_url": url})
    assert response.json()["status_code"] != 0
    assert response.json()["segment_count"] is None


def test_url_shapes():
    """ End-to-end test for the URL endpoint. Attempts to give a URL of an image without anything to segment. """
    url = "https://upload.wikimedia.org/wikipedia/en/9/95/Test_image.jpg"
    response = client.post("/api/segment_url", json={"image_url": url})
    assert response.json()["status_code"] == 0
    assert response.json()["segment_count"] == 0


def test_url_newspaper():
    """ End-to-end test for the URL endpoint. Attempts to give a URL of an actual newspaper with segments. """
    url = "https://www.theguardian.pe.ca/media/photologue/photos/cache/TG-web-14062018-Guardian_past-sb_large.jpg"
    response = client.post("/api/segment_url", json={"image_url": url})
    assert response.json()["status_code"] == 0
    assert response.json()["segment_count"] > 0


def test_formdata_invalid_file():
    """ End-to-end test for the formdata file endpoint. Attempts to give a non-image file. """
    file_path = os.path.join(TESTS_DIR, "__init__.py")
    invalid_file = {'image_file': open(file_path, 'rb')}
    response = client.post("/api/segment_formdata", files=invalid_file)
    assert response.json()["status_code"] != 0
    assert response.json()["segment_count"] is None


def test_formdata_shapes():
    """ End-to-end test for the formdata file endpoint. Attempts to give an image file without segments. """
    shapes_path = os.path.join(TEST_ASSETS_DIR, "shapes.png")
    image_file = {'image_file': open(shapes_path, 'rb')}
    response = client.post("/api/segment_formdata", files=image_file)
    assert response.json()["status_code"] == 0
    assert response.json()["segment_count"] == 0


def test_formdata_newspaper():
    """ End-to-end test for the formdata file endpoint. Attempts to give an image file of a newspaper. """
    shapes_path = os.path.join(TEST_ASSETS_DIR, "newspaper_issue.png")
    image_file = {'image_file': open(shapes_path, 'rb')}
    response = client.post("/api/segment_formdata", files=image_file)
    assert response.json()["status_code"] == 0
    assert response.json()["segment_count"] > 0
