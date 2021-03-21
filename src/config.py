import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(SCRIPT_DIR, "resources")
MAX_IMAGE_SIZE = 3840 #Images get rescaled to this resolution when running through the pipeline
MINIMUM_CONFIDENCE_THRESHOLD = 0.666 #Extracted segments with a confidence less than this are not included in the response
IMAGE_DOWNLOAD_TIMEOUT = 5 #How many seconds to wait for any response when downloading an image.
URL_REGEX = re.compile(
    r'^(?:http|ftp)s?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'
    r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)
