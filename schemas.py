from pydantic import BaseModel
from PIL import Image
from typing import List
from enum import Enum


class BoundingBox(BaseModel):
    """ Data wrapper for bounding box coordinates. """
    upper_left_x: int
    upper_left_y: int
    lower_right_x: int
    lower_right_y: int

    def __getitem__(self, i):
        if i < 0 or i > 3:
            raise Exception(f"Box does not have index {i}")
        if i == 0:
            return self.upper_left_x
        if i == 1:
            return self.upper_left_y
        if i == 2:
            return self.lower_right_x
        if i == 3:
            return self.lower_right_y



# Object type enum

class Categories(Enum):
    illustration = 'illustration'
    photograph = "photograph"
    comic_cartoon = "comics/cartoon"
    editorial_cartoon = "editorial_cartoon"
    map = "map"
    headline = "headline"
    ad = "ad"

# Segments extracted from article image

class ExtractedSegment(BaseModel):
    image: Image.Image
    text: str
    box: Box
    embeddings: List[int]
    category: Categories
    confidence: float
    parentID: str

# Original newspaper extract

class Article(BaseModel):
    image: Image.Image
    metadata: dict