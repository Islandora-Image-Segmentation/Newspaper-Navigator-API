""" This module contains code for loading and fetching the pretrained 
    PyTorch model weights from disk. """

import os

import detectron2


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "resources")
MODEL_WEIGHTS_PATH = os.path.join(RESOURCES_DIRECTORY, "model_final.pth")
MODEL_CONFIG_PATH = os.path.join(RESOURCES_DIRECTORY, "faster_rcnn_R_50_FPN_3x.yaml")
_model = None


def get_model():
    if _model is None:
        _model = load_model_from_disk()
    return _model


def load_model_from_disk():
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model_config = build_detectron_config()
        model = detectron2.build_model(model_config)
        detectron2.DetectionCheckpointer(model).load(MODEL_WEIGHTS_PATH)
        model.train(False)
        return model
    else:
        raise Exception(f"Could not find the model weights at {MODEL_WEIGHTS_PATH}. Please follow installation instructions to download the weights.")


def build_detectron_config():
    if os.path.exists(MODEL_CONFIG_PATH):
        model_config = detectron2.get_cfg()
        model_config.merge_from_file(MODEL_CONFIG_PATH)
        model_config.MODEL.ROI_HEADS.NUM_CLASSES = 7
        return model_config
    else:
        raise Exception(f"Could not find the model config at {MODEL_CONFIG_PATH}. Please follow the installation instructions to download the config.")