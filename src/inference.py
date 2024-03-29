import logging
import numpy as np
import torch
import utils
from PIL import Image
from model import get_inference_model
from schemas import ModelOutput, BoundingBox
from typing import Dict


CATEGORIES = {0: "Photograph",
              1: "Illustration",
              2: "Map",
              3: "Comics/Cartoons",
              4: "Editorial Cartoon",
              5: "Headline",
              6: "Advertisement"}


def image_to_model_input(image: Image.Image) -> Dict[str, torch.Tensor]:
    """ Converts a PIL image to the format that the ML model expects. 
    Input:
        image: A PIL image
    Output:
        A dictionary with an "image" key, and a tensor representation of the image.
    """
    standardized_image = utils.standardize_image(image)
    image_array = np.asarray(standardized_image)
    image_array = np.transpose(image_array, (2, 0, 1)) #Swap RGB channels because model expects RBG, but PIL is RGB
    return {"image": torch.from_numpy(image_array.copy())}


def predict(image: Image.Image):
    """ Take an image and run it through the inference model. This returns a ModelOutput object with all of the 
    information that the model returns. Furthermore, bounding box coordinates are normalized.
    """
    logging.debug("Sending image to model for inference ...")
    width, height = image.size
    model = get_inference_model()
    model.eval()
    with torch.no_grad():
        model_input = image_to_model_input(image)
        model_output = model([model_input])[0]
        logging.debug(f"Model returned {len(model_output)} fields.")

        bounding_boxes = model_output["instances"].get_fields()["pred_boxes"].to("cpu").tensor.tolist()
        confidences = model_output["instances"].get_fields()["scores"].to("cpu").tolist()
        classes = model_output["instances"].get_fields()["pred_classes"].to("cpu").tolist()
        classes = [CATEGORIES[num] for num in classes]

        # Normalize all the bounding box coordinates to between 0 and 1.
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
