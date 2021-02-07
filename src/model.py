""" This module contains code for loading and fetching the pretrained 
    PyTorch model weights from disk. """

import os

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from img2vec_pytorch import Img2Vec


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "resources")
INFERENCE_MODEL_WEIGHTS = os.path.join(RESOURCES_DIRECTORY, "model_final.pth")
_inference_model = None
_embedding_model = None


def get_inference_model():
    global _inference_model
    if _inference_model is None:
        _inference_model = load_inference_model_from_disk()
    return _inference_model


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = Img2Vec(cuda=False, model='resnet-18') #Smaller model runs faster on CPU
    return _embedding_model


def load_inference_model_from_disk():
    if os.path.exists(INFERENCE_MODEL_WEIGHTS):
        model_config = build_detectron_config()
        model = detectron2.modeling.build_model(model_config)
        detectron2.checkpoint.DetectionCheckpointer(model).load(INFERENCE_MODEL_WEIGHTS)
        model.train(False)
        return model
    else:
        raise Exception(f"Could not find the model weights at {INFERENCE_MODEL_WEIGHTS}. Please follow installation instructions to download the weights.")


def build_detectron_config():
    model_config = get_cfg()
    model_config.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    model_config.MODEL.ROI_HEADS.NUM_CLASSES = 7
    model_config.MODEL.DEVICE = 'cpu'
    return model_config
