from pydantic import BaseModel
from PIL import Image
from typing import List
from enum import Enum


class SegmentationRequest(BaseModel):
    image_base64: str


class SegmentationResponse(BaseModel):
    pass


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
    
# Original newspaper extract


class Article(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    image: Image.Image
    metadata: dict

# Segments extracted from article image

class ExtractedSegment(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    image: Image.Image
    text: str
    box: BoundingBox
    embeddings: List[int]
    category: Categories
    confidence: float
    parent_ref: Article
