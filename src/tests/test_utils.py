from PIL import Image
from utils import crop
import os

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to test_assets
TEST_ASSETS = os.path.join(CURRENT_SCRIPT_DIR, "test_assets")

"""First crop function test"""

def test_crop_one():
  #Load first Image
  img = Image.open(os.path.join(TEST_ASSETS, "test_image_one.png"))

  #Extract width and height of original image
  initial_width, initial_height = img.size

  #Call crop function from utils
  img_cropped = crop(img, [0.5, 0.5, 1, 1])

  #Extract width and height of cropped image
  final_width, final_height = img_cropped.size

  #Check if crop function returns a null object
  assert img_cropped is not None

  #Check if cropped image has reduced in size
  assert (initial_width >= final_width and initial_height > final_height) \
         or (initial_width > final_width and initial_height >= final_height)

"""Second crop function test"""

def test_crop_two():
  #Load second Image
  img = Image.open(os.path.join(TEST_ASSETS, "test_image_two.png"))

  #Extract width and height of original image
  initial_width, initial_height = img.size

  #Call crop function from utils
  img_cropped = crop(img, [0.5, 0.5, 1, 1])

  #Extract width and height of cropped image
  final_width, final_height = img_cropped.size

  #Check if crop function returns a null object
  assert img_cropped is not None

  #Check if cropped image has reduced in size
  assert (initial_width >= final_width and initial_height > final_height) \
         or (initial_width > final_width and initial_height >= final_height)
