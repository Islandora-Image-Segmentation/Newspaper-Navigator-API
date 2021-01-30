from PIL import Image

def crop(image: Image.Image, box) -> Image.Image:
  """ Use built in crop function for PIL Image. Normalized Box co-ordinates converted to image co-ordinates. """
  cropped = image.crop((box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height))

  return cropped