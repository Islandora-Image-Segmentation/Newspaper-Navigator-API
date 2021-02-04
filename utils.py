from PIL import Image

import config


def crop(image: Image.Image, box) -> Image.Image:
  """ Use built in crop function for PIL Image. Normalized Box co-ordinates converted to image co-ordinates. """
  return image.crop((box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height))



def standardize_image(image: Image.Image) -> Image.Image:
  """ Standardize image to RGB and max width/height of 3000 (while maintaining aspect ratio) """
  standardized_image = image.convert("RGB")
  if max(image.size) > config.MAX_IMAGE_SIZE:
      standardized_image.thumbnail((config.MAX_IMAGE_SIZE, config.MAX_IMAGE_SIZE), Image.ANTIALIAS)
  return standardized_image

  