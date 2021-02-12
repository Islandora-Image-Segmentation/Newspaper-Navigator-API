import io
import base64
import re

import requests
import PIL

import config


def crop(image: PIL.Image.Image, box) -> PIL.Image.Image:
  """Crop the image according to box, which should contain normalized coordinates."""
  return image.crop((box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height))


def standardize_image(image: PIL.Image.Image) -> PIL.Image.Image:
  """ Standardize image to RGB and max width/height of 3000 (while maintaining aspect ratio) """
  standardized_image = image.convert("RGB")
  if max(image.size) > config.MAX_IMAGE_SIZE:
      standardized_image.thumbnail((config.MAX_IMAGE_SIZE, config.MAX_IMAGE_SIZE), PIL.Image.ANTIALIAS)
  return standardized_image


def image_to_base64(image: PIL.Image.Image, image_format="JPEG2000") -> str:
  memory_buffer = io.BytesIO()
  image.save(memory_buffer, format=image_format)
  return base64.b64encode(memory_buffer.getvalue()).decode("ascii")


def base64_to_image(base64_string: str) -> PIL.Image.Image:
  image_bytes = base64.b64decode(base64_string)
  return PIL.Image.open(io.BytesIO(image_bytes))


def download_image(image_url: str) -> PIL.Image.Image:
  assert re.match(config.URL_REGEX, image_url) is not None #confirm that it's a valid URL
  response = requests.get(image_url,
                          verify=False,
                          timeout=config.IMAGE_DOWNLOAD_TIMEOUT)
  if response.status_code == 200:
      image_bytes = io.BytesIO(response.content)
      return PIL.Image.open(image_bytes)
  else:
      raise Exception(f"Received status code {response.status_code} when downloading file {image_url}.")
