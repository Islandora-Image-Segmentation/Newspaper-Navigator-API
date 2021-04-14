import os
import pytest
from PIL import Image
from utils import crop

from . import TEST_ASSETS_DIR


@pytest.mark.parametrize("test_image", ["test_image_one.png", "test_image_two.png"])
def test_crop(test_image):
    """ Test for the crop function which accepts normalized bounding boxes. """
    img = Image.open(os.path.join(TEST_ASSETS_DIR, "test_image_one.png"))
    initial_width, initial_height = img.size
    img_cropped = crop(img, [0.5, 0.5, 1, 1])
    final_width, final_height = img_cropped.size
    assert img_cropped is not None
    assert (initial_width >= final_width and initial_height > final_height) \
           or (initial_width > final_width and initial_height >= final_height)
