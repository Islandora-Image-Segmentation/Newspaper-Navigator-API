import os
from PIL import Image
from inference import predict

from . import TEST_ASSETS_DIR


def test_inference_one():
    # Load first Image
    img = Image.open(os.path.join(TEST_ASSETS_DIR, "test_image_one.png"))

    # Call predict from inference.py
    result = predict(img)

    # Number of bounding boxes should be greater than 0
    assert len(result.bounding_boxes) > 0


"""Second inference module test"""


def test_inference_two():
    # Load second Image
    img = Image.open(os.path.join(TEST_ASSETS_DIR, "test_image_two.png"))

    # Call predict from inference.py
    result = predict(img)

    # Number of bounding boxes should be greater than 0
    assert len(result.bounding_boxes) > 0
