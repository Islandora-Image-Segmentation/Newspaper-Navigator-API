import base64
import config
import io
import re
import logging 

import requests
from PIL import Image


FILE_DOWNLOAD_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Apple'
                  'WebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/74.0.3729.157 Safari/537.36'}


def crop(image: Image.Image, box) -> Image.Image:
    """Crop the image according to box, which should contain normalized coordinates."""
    return image.crop((box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height))


def standardize_image(image: Image.Image) -> Image.Image:
    """ Standardize image to RGB and max width/height of 3000 (while maintaining aspect ratio) """
    logging.debug("Standardizing image ...")
    standardized_image = image.convert("RGB")
    if max(image.size) > config.MAX_IMAGE_SIZE:
        standardized_image.thumbnail((config.MAX_IMAGE_SIZE, config.MAX_IMAGE_SIZE), Image.ANTIALIAS)
    return standardized_image


def image_to_base64(image: Image.Image, image_format="JPEG2000") -> str:
    memory_buffer = io.BytesIO()
    image.save(memory_buffer, format=image_format)
    return base64.b64encode(memory_buffer.getvalue()).decode("ascii")


def base64_to_image(base64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))


def download_image(image_url: str) -> Image.Image:
    logging.debug(f"Verifying that URL is valid: {image_url}")
    assert re.match(config.URL_REGEX, image_url) is not None  # confirm that it's a valid URL
    logging.debug(f"Trying to download image...")
    response = requests.get(image_url,
                            verify=False,
                            timeout=config.IMAGE_DOWNLOAD_TIMEOUT,
                            headers=FILE_DOWNLOAD_HEADERS)
    if response.status_code == 200:
        logging.debug(f"Successfully downloaded file at URL. Converting to image...")
        image_bytes = io.BytesIO(response.content)
        return Image.open(image_bytes)
    else:
        logging.error(f"Received status code {response.status_code} when downloading image from {image_url}")
        raise Exception(f"Received status code {response.status_code} when downloading image from {image_url}.")
