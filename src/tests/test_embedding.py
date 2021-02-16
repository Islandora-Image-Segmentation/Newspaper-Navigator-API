import os
import pytest
from PIL import Image
from embedding import generate_embeddings

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to test_assets
TEST_ASSETS = os.path.join(CURRENT_SCRIPT_DIR, "test_assets")

"""Embedding function test"""


@pytest.mark.parametrize("test_image", ["test_image_one.png", "test_image_two.png"])
def test_embedding(test_image):
    # Load Image
    img = Image.open(os.path.join(TEST_ASSETS, test_image)).convert('RGB')

    # Call generate_embeddings from embedding.py
    embedding = generate_embeddings(img)

    # Check to see if function returns non-empty list
    assert len(embedding) > 0
