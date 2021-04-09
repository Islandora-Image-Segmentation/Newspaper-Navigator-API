import os
from PIL import Image
from inference import predict

from . import TEST_ASSETS_DIR


def test_inference_one():
    """ Test for the segmentation ML model. 
    This test requires the model weights `model_final.pth` to be present in src/resources. 
    """
    img = Image.open(os.path.join(TEST_ASSETS_DIR, "test_image_one.png"))
    result = predict(img)
    assert len(result.bounding_boxes) > 0


def test_inference_two():
    """ Test for the segmentation ML model.
    This test requires the model weights `model_final.pth` to be present in src/resources. 
    """
    img = Image.open(os.path.join(TEST_ASSETS_DIR, "test_image_two.png"))
    result = predict(img)
    assert len(result.bounding_boxes) > 0
