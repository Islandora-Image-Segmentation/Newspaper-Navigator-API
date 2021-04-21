""" This module is the main entry point for the API.
    It contains all of the endpoint definitions and the CLI.
    The API can be launched by executing this module. See the README for more information.
"""

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
    """ This function gets called whenever an endpoint encounters an error. 
        Input:
            e: The error that was raised.
    """
    logging.error(f"Endpoint raised an error: {traceback.format_exc()}")
    return schemas.SegmentationResponse(status_code=-1,
                                        error_message=f"Failed to process request due to {traceback.format_exc()}",
                                        segment_count=None,
                                        segments=None)


async def validate_api_key(api_key_header: str = Security(X_API_KEY)):
    """ This function gets called for every request to validate the API key in the request's header. 
        If the API key is not what is expected, an HTTP 401 is returned. 
    """
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
    """ This endpoint accepts an image file as formdata, and returns a SegmentationResponse. """
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
    """ This endpoint accept the URL of an image, and returns a SegmentationResponse.
        The endpoint will try to download the image at the given URL.
        Note: not all servers allow for non-browser user agents to download images.
    """
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
    """ This endpoint accepts an image encoded as a base64 string, and returns a SegmentationResponse.
        Not suitable for very large images, since encoding/decoding adds computation time. 
    """
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
    """ This utility method returns a pre-calculated SegmentationResponse. 
        This pre-calculated response is used in the test endpoints. 
    """
    logging.info("Returning canned response...")
    import pickle
    pickle_path = os.path.join(config.RESOURCES_DIR, "response.pkl")
    return pickle.load(open(pickle_path, "rb"))


@app.post("/test/segment_base64", dependencies=[Security(validate_api_key)])
async def test_segment_base64(request: schemas.Base64SegmentationRequest) -> schemas.SegmentationResponse:
    """ This endpoint accepts the same input as the regular base64 one.
        However, it immediately returns a pre-calculated SegmentationResponse without doing any computation.
        Useful for testing. 
    """
    try:
        return get_canned_response()
    except Exception as e:
        return error_response(e)


@app.post("/test/segment_url", dependencies=[Security(validate_api_key)])
async def test_segment_url(request: schemas.UrlSegmentationRequest) -> schemas.SegmentationResponse:
    """ This endpoint accepts the same input as the regular URL one.
        However, it immediately returns a pre-calculated SegmentationResponse without doing any computation.
        Useful for testing. 
    """
    try:
        return get_canned_response()
    except Exception as e:
        return error_response(e)


@app.post("/test/segment_formdata", dependencies=[Security(validate_api_key)])
async def test_segment_formdata(image_file: bytes = File(...)) -> schemas.SegmentationResponse:
    """ This endpoint accepts the same input as the regular formdata one.
        However, it immediately returns a pre-calculated SegmentationResponse without doing any computation.
        Useful for testing. 
    """
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

    # Confirm that the entered log level is valid, and set up logging
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
                log_config=None) #Use log_config=None to prevent Uvicorn from writing logs twice.
