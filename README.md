# Newspaper-Navigator-API

An API wrapper around the Newspaper Navigator machine learning pipeline for extracting visual content (such as cartoons and advertisements) from newspapers. The original newspaper navigator application can be found [here](https://github.com/LibraryOfCongress/newspaper-navigator).

This API provides endpoints that accept images. When an image is submitted to the API, it is segmented and visual content (such as cartoons, maps, or advertisements) is extracted. For more information, see the `Pipeline` section in this README.


## Installation (Bare Metal)
 1. Clone this repo.
 2. Download the pre-trained model weights from [here](https://drive.google.com/file/d/1qUu3uQ8imLGp-m4DYEY5KDCrNaaS2Pb-/view?usp=sharing).
 3. Place the model weights in the `src/resources/` folder.
 4. Install [Tesseract] (https://github.com/tesseract-ocr/tessdoc/blob/master/Installation.md) and make sure `tesseract` is on your PATH.
 5. Install [PyTorch>=1.7.1](https://pytorch.org/).
 6. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).
 6. Run `pip install -r requirements.txt` to install the required python packages.


Alternatively, you can build a Docker container that contains all of the dependencies (if whatever reason you cannot install the above).
## Installation (Docker)
 1. Clone this repo.
 2. Download the pre-trained model weights from [here](https://drive.google.com/file/d/1qUu3uQ8imLGp-m4DYEY5KDCrNaaS2Pb-/view?usp=sharing).
 3. Place the model weights in the `src/resources/` folder.
 4. Install [Docker] (https://docs.docker.com/get-docker/).
 5. Run `docker-compose build`


## Hardware Requirements
Please note that images, especially large ones, take very long to segment on CPU. It can take a couple of minutes to get a response. Furthermore, the segmentation model requires a large amount of RAM. As such, ensure your system meets the following minimum requirements:

1.CPU: At least 2 cores.
2.RAM: At least 8GB.

If running into RAM or processing time issues, lower the `MAX_IMAGE_SIZE` parameter in `config.py` to process images at a lower resolution.  


 ## Launching the API
 If running locally, you can launch the API by running `python main.py`
 If running in Docker, you can launch it by running  `docker-compose up --build`.
 
 The CLI accepts the following arguments:
 1.`--port` / `-p`: The port to launch on (default `8008`)
 2.`--host`: What host to listen on (default `0.0.0.0`)
 3.`--log-level`/`-l`: Minimum log level to use (default `info`)
 4.`--timeout`/`-t`: Timeout keep alive in seconds (default `30`)
 5.`--api_key`/`-k`: The API key to use (default `None`). If not specified, the API starts without authentication.

For example, you can do `python main.py --port 5000 --api_key "abcdef"` to launch on port 5000 with an API key of "abcdef"

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

## Supported Classes
The segmentation model is trained to classify the following visual content:
1. Illustration
2. Map
3. Comics/Cartoons
4. Editorial Cartoon
5. Headline
6. Advertisement

## Testing the API with another application
Since the segmentation model takes very long to process images on CPU, it can be cumbersome to test it when integrating it with another application. To make testing easier, the API also has the following three endpoints:

1. `/test/segment_formdata`
2. `/test/segment_url`
3. `/test/segment_base64`

These endpoints behave exactly like their `/api/` counterparts (they expect the same data formats as input, and return a `SegmentationResponse`). However, these endpoints always return the same response and do so very quickly. When testing your application, you can use these endpoints to get legitimate responses very fast instead of waiting for images to process. 

## OCR
The API uses [Tesseract](https://github.com/tesseract-ocr/tesseract) in page segmentation mode 12 to perform OCR. Both the plain text OCR and the location-aware HOCR HTML are included in the segmentation response.

## Known Issues
1. When submitting an image by URL for segmentation, the API must download the image from that URL. Depending on the server that the image is hosted on, it may reject automated attempts to fetch the image. The following download headers are used to alleviate that in some cases:
```
FILE_DOWNLOAD_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Apple'
                  'WebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/74.0.3729.157 Safari/537.36'}
```
However, keep in mind that some image URLs will get rejected. 
