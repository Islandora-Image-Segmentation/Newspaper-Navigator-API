import os
import pytest
from PIL import Image
from embedding import generate_embeddings

from . import TEST_ASSETS_DIR


@pytest.mark.parametrize("test_image", ["test_image_one.png", "test_image_two.png"])
def test_embedding(test_image):
    """ Test for embedding.py module. """
    img = Image.open(os.path.join(TEST_ASSETS_DIR, test_image)).convert('RGB')
    embedding = generate_embeddings(img)
    assert len(embedding) > 0
