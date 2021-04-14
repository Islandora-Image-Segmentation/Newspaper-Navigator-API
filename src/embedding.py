import torch
from PIL import Image
from model import get_embedding_model


def generate_embeddings(image: Image.Image):
    """ Generate Resnet-18 embeddings for image. 
    Input:
        image: A PIL image.
    Output:
        embedding: List[float] of size 512
    """
    with torch.no_grad():
        inference_model = get_embedding_model()
        embedding = inference_model.get_vec(image, tensor=False)
    return embedding
