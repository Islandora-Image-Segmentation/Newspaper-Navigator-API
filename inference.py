import os
import math
import datetime
import json

import torch
import cv2
import numpy as np
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from utils import chunk

def generate_predictions(zipped):
  # unzips packed information for process to perform predictions

  S3_SAVE_DIR = zipped[0]
  OUTPUT_SAVE_DIR = zipped[1]
  dir_name = zipped[2]
  INFERENCE_BATCH_SIZE = zipped[3]
  filepaths = zipped[4]
  ID = zipped[5]

  with torch.cuda.device(ID):

    # navigates to correct directory (process is spawned in /notebooks)

    os.chdir(S3_SAVE_DIR + dir_name)

    # sets up model for process

    setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file("../../..//detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # sets prediction score threshold - this is commented out and defaults to 0.05 in Detectron2
    # if you would like to adjust the threshold, uncomment and set to the desired value in [0, 1)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # sets number of object classes to 7
    # ("Illustration/Photograph", "Photograph", "Comics/Cartoon", "Editorial Cartoon", "Map", "Headline", "Ad")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

    # build model
    model = build_model(cfg)

    # see:  https://github.com/facebookresearch/detectron2/issues/282
    # (must load weights this way if using model)
    DetectionCheckpointer(model).load("../../model_weights/model_final.pth")
    model.train(False)

    # construct batches
    batches = chunk(filepaths, math.ceil(len(filepaths) / INFERENCE_BATCH_SIZE))

    # iterate through images
    for batch in batches:

      # sets up inputs by loading in all files in batch
      inputs = []

      # stores image dimensions
      dimensions = []

      # iterate through files in batch
      for file in batch:
        # read in image
        image = cv2.imread(file)

        # store image dimensions
        height, width, _ = image.shape
        dimensions.append([width, height])

        # perform inference on batch
        image = np.transpose(image, (2, 0, 1))
        # see https://github.com/facebookresearch/detectron2/issues/282 for in-depth description of why
        # image is loaded in this way
        image_tensor = torch.from_numpy(image)
        inputs.append({"image": image_tensor})

      # performs inference
      outputs = model(inputs)

      # saves predictions
      predictions = {}

      # iterate over images in batch and save predictions to JSON
      for i in range(0, len(batch)):

        # saves filepath in format of ChronAm file structure
        predictions["filepath"] = dir_name + "data/" + batch[i].split("data_")[1].replace(dir_name, '').replace('_',
                                                                                                                '/').replace(
          '.jpg', '.jp2')

        # parses metadata from filepath
        date_str = predictions["filepath"].split('/')[-2]
        predictions["batch"] = dir_name[:-1]
        predictions["lccn"] = predictions["filepath"].split('/')[-4]
        predictions["pub_date"] = str(datetime.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])))
        predictions["edition_seq_num"] = int(date_str[8:10])
        if "a" in predictions["filepath"].split('/')[-1][:-4] or "b" in predictions["filepath"].split('/')[-1][:-4]:
          continue
        predictions["page_seq_num"] = int(predictions["filepath"].split('/')[-1][:-4])

        # saves predictions
        # we first normalize the bounding box coordinates
        boxes = outputs[i]["instances"].get_fields()["pred_boxes"].to("cpu").tensor.tolist()
        normalized_boxes = []
        width = dimensions[i][0]
        height = dimensions[i][1]

        for box in boxes:
          normalized_box = (
          box[0] / float(width), box[1] / float(height), box[2] / float(width), box[3] / float(height))
          normalized_boxes.append(normalized_box)

        # saves additional outputs of predictions
        predictions["boxes"] = normalized_boxes
        predictions["scores"] = outputs[i]["instances"].get_fields()["scores"].to("cpu").tolist()
        predictions["pred_classes"] = outputs[i]["instances"].get_fields()["pred_classes"].to("cpu").tolist()

        with open(OUTPUT_SAVE_DIR + dir_name + batch[i].replace('.jpg', '.json'), "w") as fp:
          json.dump(predictions, fp)