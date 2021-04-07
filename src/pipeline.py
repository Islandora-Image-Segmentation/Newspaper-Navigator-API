import config
import embedding
import inference
import ocr
import schemas
import utils
from PIL import Image
from typing import List


def segment_image(image: Image.Image) -> List[schemas.ExtractedSegment]:
    standardized_image = utils.standardize_image(image)
    model_output = inference.predict(standardized_image)

    for i in range(len(model_output.confidences) - 1, -1,
                   -1):  # Iterate backwards here because we're removing elements as we iterate
        if model_output.confidences[i] < config.MINIMUM_CONFIDENCE_THRESHOLD:
            del model_output.confidences[i]
            del model_output.classes[i]
            del model_output.bounding_boxes[i]

    segment_images = [utils.crop(image, box).convert("RGB") for box in model_output.bounding_boxes]
    segment_ocr = [ocr.retrieve_ocr(image) for image in segment_images]
    segment_hocr = [ocr.retrieve_hocr(image) for image in segment_images]
    segment_embeddings = [embedding.generate_embeddings(image).tolist() for image in segment_images]

    segments = []
    for i in range(len(model_output.bounding_boxes)):
        segment = schemas.ExtractedSegment(ocr_text=segment_ocr[i],
                                           hocr=segment_hocr[i],
                                           bounding_box=model_output.bounding_boxes[i],
                                           embedding=segment_embeddings[i],
                                           classification=model_output.classes[i],
                                           confidence=model_output.confidences[i])
        segments.append(segment)

    return segments
