from ocr import retrieve_ocr
from PIL import Image
import os

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to test_assets
TEST_ASSETS = os.path.join(CURRENT_SCRIPT_DIR, "test_assets")

"""First OCR function test"""

def test_ocr_one():
  #Load first Image
  img = Image.open(os.path.join(TEST_ASSETS, "test_image_one.png"))

  #Call retrieve_ocr from ocr.py
  text = retrieve_ocr(img)

  #text from first image should contain the below portions of text
  assert "This is a lot of 12 point text" in text
  assert "The quick brown dog" in text
  assert "over the lazy fox" in text

"""Second OCR function test"""

def test_ocr_two():
  #Load first Image
  img = Image.open(os.path.join(TEST_ASSETS, "test_image_two.png"))

  #Call retrieve_ocr from ocr.py
  text = retrieve_ocr(img)

  #text from first image should contain the below portions of text
  assert "to test" in text
  assert "Tesseract OCR" in text