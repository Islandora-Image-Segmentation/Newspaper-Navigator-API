from PIL import Image
from pytesseract import image_to_string, image_to_pdf_or_hocr

def retrieve_ocr(image: Image.Image) -> str:
    """ Extract string from image using pytesseract """
    return image_to_string(image, config='--psm 12')

def retrieve_hocr(image: Image.Image) -> bytes:
    """ Extract string from image using pytesseract """
    return image_to_pdf_or_hocr(image, config='--psm 12', extension="hocr")
