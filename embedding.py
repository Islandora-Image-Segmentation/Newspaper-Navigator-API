import PIL

import torch
from model import get_embedding_model        


def generate_embeddings(image: PIL.Image.Image):
    with torch.no_grad():
        inference_model = get_embedding_model()
        embedding = inference_model.get_vec(image, tensor=False)
    return embedding