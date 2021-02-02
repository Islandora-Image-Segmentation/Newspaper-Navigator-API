from PIL import Image
from utils import crop
from ocr import retrieve_ocr

"""Test class for crop function in utils"""

class TestCrop:

  """Test function using first image"""
  def test_crop_one(self):
    """Load first Image"""
    img = Image.open("tests/test_assets/test_image_one.jpg")

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
      img = Image.open("tests/test_assets/test_image_two.jpg")

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
      img = Image.open("tests/test_assets/test_image_one.jpg")

      """Call retrieve_ocr from ocr.py"""
      text = retrieve_ocr(img)

      """text from first image should be 'great!'"""
      assert text is "great!"
