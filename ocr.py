from PIL import Image
from pytesseract import image_to_string, image_to_pdf_or_hocr


def retrieve_ocr(image: Image.Image) -> dict:
  """ Extract string and HOCR from image using pytesseract """
  return {"string": image_to_string(image), "hocr": image_to_pdf_or_hocr(image, extension='hocr')}
