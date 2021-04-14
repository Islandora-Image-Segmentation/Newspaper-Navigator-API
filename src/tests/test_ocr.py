import os
import pytest
from PIL import Image
from ocr import retrieve_ocr, retrieve_hocr

from . import TEST_ASSETS_DIR


def get_ocr_test_data():
    return [('test_image_one.png', "This is a lot of 12 point text"), ('test_image_one.png', "The quick brown dog"),
            ('test_image_one.png', "over the lazy fox"), ('test_image_two.png', "Noisy"),
            ('test_image_two.png', "image"),
            ('test_image_two.png', "to test"),
            ('test_image_two.png', "Tesseract OCR")]


@pytest.mark.parametrize("test_image, test_text", get_ocr_test_data())
def test_ocr(test_image, test_text):
    """ Test for the OCR module. Gives various images and confirms that OCR'd text is as expected. """
    img = Image.open(os.path.join(TEST_ASSETS_DIR, test_image))
    text = retrieve_ocr(img)

    assert test_text[0] in text
    assert test_text[1] in text
    assert test_text[2] in text

    text = retrieve_ocr(img)
    assert test_text in text


@pytest.mark.parametrize("image_file, expected", [
    ("test_image_one.png", ["<!DOCTYPE html", "This", "brown", "fox"]),
    ("test_image_two.png", ["<!DOCTYPE html", "test", "Tesseract"]),
])
def test_hocr(image_file, expected):
    """ Test for the HOCR module. Gives various images and confirms that OCR'd text is as expected. """
    img = Image.open(os.path.join(TEST_ASSETS_DIR, image_file))
    text = retrieve_hocr(img).decode('utf-8')
    for exp in expected:
        assert exp in text
