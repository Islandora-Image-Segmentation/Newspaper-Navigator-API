from PIL import Image
from pytesseract import image_to_string

def retrieve_ocr(image: Image.Image) -> str:
  """ Extract string from image using pytesseract """
  return image_to_string(image, config='--psm 12')
