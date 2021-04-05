from PIL import Image
from enum import Enum
from pydantic import BaseModel
from typing import List
from typing import Optional


class BoundingBox(BaseModel):
    """ Data wrapper for bounding box coordinates. """
    upper_left_x: float
    upper_left_y: float
    lower_right_x: float
    lower_right_y: float

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


class Categories(Enum):
    illustration = 'illustration'
    photograph = "photograph"
    comic_cartoon = "comics/cartoon"
    editorial_cartoon = "editorial_cartoon"
    map = "map"
    headline = "headline"
    ad = "ad"


class Article(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    image: Image.Image
    metadata: dict


class ExtractedSegment(BaseModel):
    ocr_text: str
    hocr: str
    bounding_box: BoundingBox
    embedding: List[float]
    classification: str
    confidence: float


class ModelOutput(BaseModel):
    bounding_boxes: List[BoundingBox]
    confidences: List[float]
    classes: List[str]


class UrlSegmentationRequest(BaseModel):
    image_url: str


class Base64SegmentationRequest(BaseModel):
    image_base64: str


class SegmentationResponse(BaseModel):
    status_code: int
    error_message: str
    segment_count: Optional[int]
    segments: Optional[List[ExtractedSegment]]
