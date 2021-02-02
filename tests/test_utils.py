from PIL import Image
from utils import crop
from ocr import retrieve_ocr

"""Test class for crop function in utils"""

class TestCrop:

  """Test function using first image"""
  def test_crop_one(self):
    """Load first Image"""
    img = Image.open("tests/test_assets/test_image_one.png")

    """Extract width and height of original image"""
    initial_width, initial_height = img.size

    """Call crop function from utils"""
    img_cropped = crop(img, [0.5,0.5,1,1])

    """Extract width and height of cropped image"""
    final_width, final_height = img_cropped.size

    """Check if crop function returns a null object"""
    assert img_cropped is not None

    """Check if cropped image has reduced in size"""
    assert (initial_width >= final_width and initial_height > final_height) \
           or (initial_width > final_width and initial_height >= final_height)

    """Test function using second image"""

  def test_crop_two(self):
      """Load second Image"""
      img = Image.open("tests/test_assets/test_image_two.png")

      """Extract width and height of original image"""
      initial_width, initial_height = img.size

      """Call crop function from utils"""
      img_cropped = crop(img, [0.5, 0.5, 1, 1])

      """Extract width and height of cropped image"""
      final_width, final_height = img_cropped.size

      """Check if crop function returns a null object"""
      assert img_cropped is not None

      """Check if cropped image has reduced in size"""
      assert (initial_width >= final_width and initial_height > final_height) \
             or (initial_width > final_width and initial_height >= final_height)

  class TestOCRFunction:
    def test_ocr_one(self):
      """Load first Image"""
      img = Image.open("tests/test_assets/test_image_one.png")

      """Call retrieve_ocr from ocr.py"""
      text = retrieve_ocr(img)

      """text from first image should contain the below portions of text"""
      assert "This is a lot of 12 point text" in text
      assert "The quick brown dog" in text
      assert "over the lazy fox" in text

    def test_ocr_two(self):
      """Load first Image"""
      img = Image.open("tests/test_assets/test_image_two.png")

      """Call retrieve_ocr from ocr.py"""
      text = retrieve_ocr(img)

      """text from first image should contain the below portions of text"""
      assert "to test" in text
      assert "Tesseract OCR" in text


