import argparse
import re
import traceback
import io
import os

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi.security import APIKeyHeader
from fastapi import Security
from fastapi import HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED
from PIL import Image

import config
import pipeline
import schemas
import utils


app = FastAPI()

_api_key = None
X_API_KEY = APIKeyHeader(name='X-API-Key', auto_error=False)
async def validate_api_key(api_key_header: str = Security(X_API_KEY)):
    global _api_key
    if _api_key:
        if api_key_header != _api_key:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )


@app.post("/api/segment_formdata")
async def segment_formdata(image_file: bytes = File(...), dependencies=[Security(validate_api_key)]) -> schemas.SegmentationResponse:
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
async def segment_url(request: schemas.UrlSegmentationRequest, dependencies=[Security(validate_api_key)]) -> schemas.SegmentationResponse:
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


@app.post("/api/segment_base64", dependencies=[Security(validate_api_key)])
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


def get_canned_response():
    import pickle
    pickle_path = os.path.join(config.RESOURCES_DIR, "response.pkl")
    return pickle.load(open(pickle_path, "rb"))


@app.post("/test/segment_base64", dependencies=[Security(validate_api_key)])
async def test_segment_base64(request: schemas.Base64SegmentationRequest) -> schemas.SegmentationResponse:
    try:
        return get_canned_response()
    except Exception as e:
        return schemas.SegmentationResponse(status_code=-1,
                                            error_message=f"Failed to process request due to {traceback.format_exc()}",
                                            segment_count=None,
                                            segments=None)


@app.post("/test/segment_url", dependencies=[Security(validate_api_key)])
async def test_segment_url(request: schemas.UrlSegmentationRequest) -> schemas.SegmentationResponse:
    try:
        return get_canned_response()
    except Exception as e:
        return schemas.SegmentationResponse(status_code=-1,
                                            error_message=f"Failed to process request due to {traceback.format_exc()}",
                                            segment_count=None,
                                            segments=None)


@app.post("/test/segment_formdata", dependencies=[Security(validate_api_key)])
async def test_segment_formdata(image_file: bytes = File(...)) -> schemas.SegmentationResponse:
    try:
        return get_canned_response()
    except Exception as e:
        return schemas.SegmentationResponse(status_code=-1,
                                            error_message=f"Failed to process request due to {traceback.format_exc()}",
                                            segment_count=None,
                                            segments=None)

                                            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--log_level", "-l", type=str, default="info")  
    parser.add_argument("--timeout", "-t", type=int, default=30)  
    parser.add_argument("--api_key", "-k", type=str, default=None)
    args = parser.parse_args()

    if args.api_key:
        _api_key = str(args.api_key)

    uvicorn.run(app,
                port=args.port,
                host=args.host,
                log_level=args.log_level,
                timeout_keep_alive=args.timeout)
