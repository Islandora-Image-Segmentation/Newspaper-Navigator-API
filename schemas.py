from pydantic import BaseModel
from PIL import Image
from typing import List
from enum import Enum

# Bounding box coordinates

class Box(BaseModel):
    lower_left: int
    upper_left: int
    upper_right: int
    lower_right: int

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