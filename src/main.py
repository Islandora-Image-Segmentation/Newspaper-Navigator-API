import traceback
import argparse

import uvicorn
from fastapi import FastAPI

import utils
import config
import inference
import ocr
import embedding
import schemas


parser = argparse.ArgumentParser()
parser.add_argument("--port", "-p", type=int, default=8000)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--log_level", "-l", type=str, default="info") # Possible options are: critical, error, warning, info, debug, trace
parser.add_argument("--timeout", "-t", type=int, default=30) # Timeout in seconds
args = parser.parse_args()


app = FastAPI()


@app.post("/api/segment_article")
async def segment_article(request: schemas.SegmentationRequest) -> schemas.SegmentationResponse:
    try:
        image = utils.base64_to_image(request.image_base64)
        standardized_image = utils.standardize_image(image)
        model_output = inference.predict(standardized_image)

        for i in range(len(model_output.confidences)-1, -1, -1): #Iterate backwards here because we're removing elements as we iterate
            if model_output.confidences[i] < config.MINIMUM_CONFIDENCE_THRESHOLD:
                del model_output.confidences[i]
                del model_output.classes[i]
                del model_output.bounding_boxes[i]

        segment_images = [utils.crop(image, box).convert("RGB") for box in model_output.bounding_boxes]
        segment_ocr = [ocr.retrieve_ocr(image) for image in segment_images]
        segment_embeddings = [embedding.generate_embeddings(image).tolist() for image in segment_images]

        segments = []
        for i in range(len(model_output.bounding_boxes)):
            segment = schemas.ExtractedSegment(ocr_text=segment_ocr[i],
                                    bounding_box=model_output.bounding_boxes[i],
                                    embedding=segment_embeddings[i],
                                    classification=model_output.classes[i],
                                    confidence=model_output.confidences[i])
            segments.append(segment)

        return schemas.SegmentationResponse(status_code=0,
                                    error_message="",
                                    segment_count=len(segments),
                                    segments=segments)
    except Exception as e:
        return schemas.SegmentationResponse(status_code=-1,
                                    error_message=f"Failed to process request due to {traceback.format_exc()}",
                                    segment_count=None,
                                    segments=None)     

uvicorn.run(app, 
            port=args.port, 
            host=args.host, 
            log_level=args.log_level,
            timeout_keep_alive=args.timeout)
