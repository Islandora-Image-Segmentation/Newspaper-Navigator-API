# Newspaper-Navigator-API

An API wrapper around the Newspaper Navigator machine learning pipeline for extracting visual content (such as cartoons and advertisements) from newspapers.

## Installation (Bare Metal)
 1. Clone this repo.
 2. Download the pre-trained model weights from [here](https://drive.google.com/file/d/1qUu3uQ8imLGp-m4DYEY5KDCrNaaS2Pb-/view?usp=sharing).
 3. Place the model weights in the `src/resources/` folder.
 4. Install [Tesseract] (https://github.com/tesseract-ocr/tessdoc/blob/master/Installation.md) and make sure `tesseract` is on your PATH.
 5. Install [PyTorch>=1.7.1](https://pytorch.org/).
 6. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).
 6. Run `pip install -r requirements.txt` to install the required python packages.

Alternatively, you can build a Docker container containing all the dependencies (if whatever reason you cannot install the above).
## Installation (Docker)
 1. Clone this repo.
 2. Download the pre-trained model weights from [here](https://drive.google.com/file/d/1qUu3uQ8imLGp-m4DYEY5KDCrNaaS2Pb-/view?usp=sharing).
 3. Place the model weights in the `src/resources/` folder.
 4. Install [Docker] (https://docs.docker.com/get-docker/).
 5. Run `docker-compose build`


 ## Launching the API
 If running locally, you can launch the API by running `python main.py`
 If running in Docker, you can launch it by running  `docker-compose up --build`.
 
 The CLI accepts the following arguments:
 1.`--port` / `-p`: The port to launch on (default `8000`)
 2.`--host`: What host to listen on (default `0.0.0.0`)
 3.`--log-level`/`-l`: Minimum log level to use (default `info`)
 4.`--timeout`/`-t`: Timeout keep alive in seconds (default `30`)

For example, you can do `python main.py --port 5000` to launch on port 5000 instead.

## Using GPU
1. Set `USE_CPU` in `config.py` to `False`.
2. Install the latest Nvidia drivers.
3. Install CUDA toolkit.
4. Make sure your Torch version supports CUDA.
6. If using Docker, install nvidia-container-toolkit.
7. If using Docker, do `docker-compose -f docker-compose-GPU.yml up` instead of `docker-compose up`.


## Endpoints
The API has the following endpoints:

`/api/segment_formdata`: This endpoint expects the image to be segmented appended as formdata.
`/api/segment_url`: This endpoint expects a POST request with JSON body in the format of `{"image_url": URL_HERE}`
`/api/segment_base64`: This endpoint expects a POST request with JSON body in the format of `{"image_base64": BASE64_HERE}`

All endpoints will return a `SegmentationResponse` in the following format:

```
class SegmentationResponse(BaseModel):
    status_code: int
    error_message: str
    segment_count: Optional[int]
    segments: Optional[List[ExtractedSegment]]
```

where ExtractedSegment is defined by:

```
class ExtractedSegment(BaseModel):
    ocr_text: str
    hocr: str
    bounding_box: BoundingBox
    embedding: List[float]
    classification: str
    confidence: float
```

Note:  If something goes wrong with your request, the `status_code` of the response will be non-zero and the reason will be returned in `error_message`. In that case, `segment_count` and `segments` will be null so make sure to check the `status_code` of the response before accessing those fields.

## Running the tests
The API uses PyTest for testing. Tests are located in `/src/tests`.
You can run the tests by navigating to `src` and calling `pytest` in your terminal.
Note that this will require you to have already downloaded the model weights.

## Pipeline
Images go through a pipeline that can be broken into the following steps:

1. The image is given to a pretrained FasterRCNN-based model that returns bounding boxes, classifications, and confidences for visual content. 
2. All results below a configurable minimum confidence threshold are discarded.
3. The segments are cropped out from the original image.
4. Each segment goes through OCR using Tesseract.
5. Each segment goes through a pretrained Resnet18 model to generate image embeddings (useful for similarity comparison and search by image).
