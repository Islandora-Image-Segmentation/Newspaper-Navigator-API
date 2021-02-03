import io
import base64

from PIL import Image


def crop(image: Image.Image, box) -> Image.Image:
  """Crop the image according to box, which should contains normalized coordinates."""
  return image.crop((box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height))


def image_to_base64(image: Image.Image) -> str:
  memory_buffer = io.BytesIO()
  image.save(memory_buffer, format='PNG')
  return base64.b64encode(memory_buffer.getvalue()).decode('ascii')


def base64_to_image(base64_string: str) -> Image.Image:
  image_bytes = base64.b64decode(base64_string)
  return Image.open(io.BytesIO(image_bytes))
