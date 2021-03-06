import os
import pytest
from PIL import Image
from utils import crop

from . import TEST_ASSETS_DIR

@pytest.mark.parametrize("test_image", ["test_image_one.png", "test_image_two.png"])
def test_crop(test_image):
    # Load Image
    img = Image.open(os.path.join(TEST_ASSETS_DIR, "test_image_one.png"))

    # Extract width and height of original image
    initial_width, initial_height = img.size

    # Call crop function from utils
    img_cropped = crop(img, [0.5, 0.5, 1, 1])

    # Extract width and height of cropped image
    final_width, final_height = img_cropped.size

    # Check if crop function returns a null object
    assert img_cropped is not None

    # Check if cropped image has reduced in size
    assert (initial_width >= final_width and initial_height > final_height) \
           or (initial_width > final_width and initial_height >= final_height)
