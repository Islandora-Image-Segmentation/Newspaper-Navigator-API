import argparse
import config
import io
import os
import pipeline
import re
import schemas
import traceback
import utils
import uvicorn
from PIL import Image
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import Request
from fastapi import Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED
import logging

app = FastAPI()

_api_key = None
X_API_KEY = APIKeyHeader(name='X-API-KEY', auto_error=False)


def error_response(e):
    logging.error(f"Endpoint raised an error: {traceback.format_exc()}")
    return schemas.SegmentationResponse(status_code=-1,
                                            error_message=f"Failed to process request due to {traceback.format_exc()}",
                                            segment_count=None,
                                            segments=None)


async def validate_api_key(api_key_header: str = Security(X_API_KEY)):
    global _api_key
    if _api_key:
        if api_key_header != _api_key:
            logging.error(f"Received invalid API key: {api_key_header}")
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )


@app.post("/api/segment_formdata", dependencies=[Security(validate_api_key)])
async def segment_formdata(image_file: bytes = File(...)) -> schemas.SegmentationResponse:
    try:
        image = Image.open(io.BytesIO(image_file))
        segments = pipeline.segment_image(image)
        return schemas.SegmentationResponse(status_code=0,
                                            error_message="",
                                            segment_count=len(segments),
                                            segments=segments)
    except Exception as e:
        return error_response(e)
        

@app.post("/api/segment_url", dependencies=[Security(validate_api_key)])
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
        return error_response(e)


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
        return error_response(e)


def get_canned_response():
    logging.info("Returning canned response...")
    import pickle
    pickle_path = os.path.join(config.RESOURCES_DIR, "response.pkl")
    return pickle.load(open(pickle_path, "rb"))


@app.post("/test/segment_base64", dependencies=[Security(validate_api_key)])
async def test_segment_base64(request: schemas.Base64SegmentationRequest) -> schemas.SegmentationResponse:
    try:
        return get_canned_response()
    except Exception as e:
        return error_response(e)


@app.post("/test/segment_url", dependencies=[Security(validate_api_key)])
async def test_segment_url(request: schemas.UrlSegmentationRequest) -> schemas.SegmentationResponse:
    try:
        return get_canned_response()
    except Exception as e:
        return error_response(e)


@app.post("/test/segment_formdata", dependencies=[Security(validate_api_key)])
async def test_segment_formdata(image_file: bytes = File(...)) -> schemas.SegmentationResponse:
    try:
        return get_canned_response()
    except Exception as e:
        return error_response(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8008)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--log_level", "-l", type=str, default="info")
    parser.add_argument("--timeout", "-t", type=int, default=30)
    parser.add_argument("--api_key", "-k", type=str, default=None)
    args = parser.parse_args()

    if args.api_key:
        _api_key = str(args.api_key)

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % numeric_level)
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=numeric_level)
    logging.basicConfig(filename='API_logs.log', level=numeric_level)

    uvicorn.run(app,
                port=args.port,
                host=args.host,
                log_level=args.log_level,
                timeout_keep_alive=args.timeout,
                log_config=None)
