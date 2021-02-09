import pytest

from ocr import retrieve_ocr
from PIL import Image
import os

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to test_assets
TEST_ASSETS = os.path.join(CURRENT_SCRIPT_DIR, "test_assets")

"""OCR function test"""


def get_ocr_test_data():
  return [('test_image_one.png', "This is a lot of 12 point text"), ('test_image_one.png', "The quick brown dog"),
          ('test_image_one.png', "over the lazy fox"), ('test_image_two.png', "Noisy image"),
          ('test_image_two.png', "to test"),
          ('test_image_two.png', "Tesseract OCR")]

@pytest.mark.parametrize("test_image, test_text", get_ocr_test_data())
def test_ocr(test_image, test_text):
  # Load Image
  img = Image.open(os.path.join(TEST_ASSETS, test_image))

  # Call retrieve_ocr from ocr.py
  text = retrieve_ocr(img)

  # text from image should contain the below portions of text
  assert test_text[0] in text
  assert test_text[1] in text
  assert test_text[2] in text
