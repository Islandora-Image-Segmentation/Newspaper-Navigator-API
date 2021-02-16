from typing import Dict

from PIL import Image
import torch
import numpy as np

from model import get_inference_model
from schemas import ModelOutput, BoundingBox

CATEGORIES = {0: "Photograph",
              1: "Illustration",
              2: "Map",
              3: "Comics/Cartoons",
              4: "Editorial Cartoon",
              5: "Headline",
              6: "Advertisement"}


def image_to_model_input(image: PIL.Image.Image) -> Dict[str, torch.Tensor]:
    standardized_image = utils.standardize_image(image)
    image_array = np.asarray(standardized_image)
    image_array = np.transpose(image_array, (2, 0, 1))
    return {"image": torch.from_numpy(image_array.copy())}


def predict(image: Image.Image):
    width, height = image.size
    model = get_inference_model()
    model.eval()
    with torch.no_grad():
        model_input = image_to_model_input(image)
        model_output = model([model_input])[0]
        bounding_boxes = model_output["instances"].get_fields()["pred_boxes"].to("cpu").tensor.tolist()
        confidences = model_output["instances"].get_fields()["scores"].to("cpu").tolist()
        classes = model_output["instances"].get_fields()["pred_classes"].to("cpu").tolist()
        classes = [CATEGORIES[num] for num in classes]
        normalized_bounding_boxes = []
        for box in bounding_boxes:
            normalized_box = (
            box[0] / float(width), box[1] / float(height), box[2] / float(width), box[3] / float(height))
            normalized_bounding_boxes.append(BoundingBox(upper_left_x=normalized_box[0],
                                                         upper_left_y=normalized_box[1],
                                                         lower_right_x=normalized_box[2],
                                                         lower_right_y=normalized_box[3]))
        return ModelOutput(bounding_boxes=normalized_bounding_boxes,
                           confidences=confidences,
                           classes=classes)
