import detectron2
import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from img2vec_pytorch import Img2Vec

import config


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "resources")
INFERENCE_MODEL_WEIGHTS = os.path.join(RESOURCES_DIRECTORY, "model_final.pth")
_inference_model = None
_embedding_model = None


def get_inference_model():
    """ Lazily load the inference model from disk, but cache a reference to it once it has been loaded."""
    global _inference_model
    if _inference_model is None:
        _inference_model = load_inference_model_from_disk()
    return _inference_model


def get_embedding_model(use_cpu=config.USE_CPU):
    """ Lazily load the embedding model from disk, but cache a reference to it once it has been loaded."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = Img2Vec(cuda=not use_cpu, model='resnet-18')
    return _embedding_model
    

def load_inference_model_from_disk():
    """ This function fetches, configured, and returns the inference model from disk.
        The model weights are expected to be in src/resources. 
    """
    if os.path.exists(INFERENCE_MODEL_WEIGHTS):
        model_config = build_detectron_config()
        model = detectron2.modeling.build_model(model_config)
        detectron2.checkpoint.DetectionCheckpointer(model).load(INFERENCE_MODEL_WEIGHTS)
        model.train(False)
        return model
    else:
        raise Exception(
            f"Could not find the model weights at {INFERENCE_MODEL_WEIGHTS}. Please follow installation instructions to download the weights.")


def build_detectron_config(use_cpu=config.USE_CPU):
	""" This utility function builds and returns a Detectron config object. """
    model_config = get_cfg()
    model_config.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    model_config.MODEL.ROI_HEADS.NUM_CLASSES = 7
    if use_cpu:
        model_config.MODEL.DEVICE = 'cpu'
    return model_config
