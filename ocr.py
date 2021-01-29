from PIL import Image
import pytesseract
from pytesseract import image_to_string

def retrieve_ocr(image: Image.Image):
  # Tell pytesseract where it's stored on windows devices, not necessary for linux or unix based systems
  pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

  # Extract string from image using pytesseract
  return image_to_string(image)