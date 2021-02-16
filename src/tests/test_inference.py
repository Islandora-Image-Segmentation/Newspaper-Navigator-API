from inference import predict
from PIL import Image
import os

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to test_assets
TEST_ASSETS = os.path.join(CURRENT_SCRIPT_DIR, "test_assets")

"""First inference module test"""


def test_inference_one():
    # Load first Image
    img = Image.open(os.path.join(TEST_ASSETS, "test_image_one.png"))

    # Call predict from inference.py
    result = predict(img)

    # Number of bounding boxes should be greater than 0
    assert len(result.bounding_boxes) > 0


"""Second inference module test"""


def test_inference_two():
    # Load second Image
    img = Image.open(os.path.join(TEST_ASSETS, "test_image_two.png"))

    # Call predict from inference.py
    result = predict(img)

    # Number of bounding boxes should be greater than 0
    assert len(result.bounding_boxes) > 0
