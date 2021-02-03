from typing import Dict

import PIL
import torch
import numpy as np

from model import get_inference_model
from schemas import ModelOutput


def image_to_model_input(image: PIL.Image.Image) -> Dict[str, torch.Tensor]:
    image_array = np.asarray(image)
    image_array = np.transpose(image_array, (2, 0, 1))
    return {"image": torch.from_numpy(image_array.copy())}


def predict(image: PIL.Image.Image):
    width, height = image.size
    model = get_inference_model()
    model.eval()
    with torch.no_grad():
        model_input = image_to_model_input(image)
        model_output = model([model_input])

        bounding_boxes = model_output["instances"].get_fields()["pred_boxes"].to("cpu").tensor.tolist()
        confidences = model_output["instances"].get_fields()["scores"].to("cpu").tolist()
        classes = model_output["instances"].get_fields()["pred_classes"].to("cpu").tolist()
        
        normalized_bounding_boxes = []
        for box in bounding_boxes:
            normalized_box = (box[0]/float(width), box[1]/float(height), box[2]/float(width), box[3]/float(height))
            normalized_bounding_boxes.append(normalized_box)

        return ModelOutput(bounding_boxes=bounding_boxes,
                           confidences=confidences,
                           classes=classes)