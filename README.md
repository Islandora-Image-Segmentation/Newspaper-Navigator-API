# Newspaper-Navigator-API

An API wrapper around the Newspaper Navigator app.

## Installation (Bare Metal)
 1. Clone this repo.
 2. Download the pre-trained model weights from [here](https://drive.google.com/file/d/1qUu3uQ8imLGp-m4DYEY5KDCrNaaS2Pb-/view?usp=sharing).
 3. Place the model weights in the `src/resources/` folder.
 4. Install [Tesseract] (https://github.com/tesseract-ocr/tessdoc/blob/master/Installation.md).
 5. Install [PyTorch>=1.7.1](https://pytorch.org/).
 6. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).
 6. Run `pip install -r requirements.txt` to install the required python packages.

## Installation (Docker)
 1. Clone this repo.
 2. Download the pre-trained model weights from [here](https://drive.google.com/file/d/1qUu3uQ8imLGp-m4DYEY5KDCrNaaS2Pb-/view?usp=sharing).
 3. Place the model weights in the `src/resources/` folder.
 4. Install [Docker] (https://docs.docker.com/get-docker/).
 5. Run `docker-compose build`

 ## Launching the API
 You can launch the API by running `python main.py` (if bare metal) or `docker-compose up --build` (if in Docker).
 
 The CLI accepts the following arguments:
 1.`--port` / `-p`: The port to launch on (default `8000`)
 2.`--host`: What host to listen on (default `0.0.0.0`)
 3.`--log-level`/`-l`: Minimum log level to use (default `info`)
 4.`--timeout`/`-t`: Timeout keep alive in seconds (default `30`)

