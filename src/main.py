import argparse
import re
import traceback
import io

import uvicorn
from fastapi import FastAPI
from fastapi import File
from PIL import Image

import config
import pipeline
import schemas
import utils

app = FastAPI()


@app.post("/api/segment_formdata")
async def segment_formdata(image_file: bytes = File(...)) -> schemas.SegmentationResponse:
    try:
        image = Image.open(io.BytesIO(image_file))
        segments = pipeline.segment_image(image)
        return schemas.SegmentationResponse(status_code=0,
                                            error_message="",
                                            segment_count=len(segments),
                                            segments=segments)
    except Exception as e:
        return schemas.SegmentationResponse(status_code=-1,
                                            error_message=f"Failed to process request due to {traceback.format_exc()}",
                                            segment_count=None,
                                            segments=None)


@app.post("/api/segment_url")
async def segment_url(request: schemas.UrlSegmentationRequest) -> schemas.SegmentationResponse:
    try:
        assert re.match(config.URL_REGEX, request.image_url)
        image = utils.download_image(request.image_url)
        segments = pipeline.segment_image(image)
        return schemas.SegmentationResponse(status_code=0,
                                            error_message="",
                                            segment_count=len(segments),
                                            segments=segments)
    except Exception as e:
        return schemas.SegmentationResponse(status_code=-1,
                                            error_message=f"Failed to process request due to {traceback.format_exc()}",
                                            segment_count=None,
                                            segments=None)


@app.post("/api/segment_base64")
async def segment_base64(request: schemas.Base64SegmentationRequest) -> schemas.SegmentationResponse:
    try:
        image = utils.base64_to_image(request.image_base64)
        segments = pipeline.segment_image(image)
        return schemas.SegmentationResponse(status_code=0,
                                            error_message="",
                                            segment_count=len(segments),
                                            segments=segments)
    except Exception as e:
        return schemas.SegmentationResponse(status_code=-1,
                                            error_message=f"Failed to process request due to {traceback.format_exc()}",
                                            segment_count=None,
                                            segments=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--log_level", "-l", type=str,
                        default="info")  # Possible options are: critical, error, warning, info, debug, trace
    parser.add_argument("--timeout", "-t", type=int, default=30)  # Timeout in seconds
    args = parser.parse_args()

    uvicorn.run(app,
                port=args.port,
                host=args.host,
                log_level=args.log_level,
                timeout_keep_alive=args.timeout)
