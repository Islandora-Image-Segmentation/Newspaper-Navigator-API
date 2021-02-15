import io
import base64

from PIL import Image

import config


def crop(image: Image.Image, box) -> Image.Image:
    """Crop the image according to box, which should contain normalized coordinates."""
    return image.crop((box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height))


def standardize_image(image: Image.Image) -> Image.Image:
    """ Standardize image to RGB and max width/height of 3000 (while maintaining aspect ratio) """
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
